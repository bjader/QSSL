#!/usr/bin/env python
"""Run linear probe experiment to evaluate self-supervised feature quality.

Example:
python linear_probe.py \
    --pretrained model/selfsup/resnet18-bsize_512-checkpoint_0020.path.tar \
    --gpu 0 -a resnet18

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
import argparse
import os
import random
import shutil
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import hybrid_resnet
import moco.builder

# Imports for yaspi
import sys
import json
import getpass
from pathlib import Path
import itertools
from yaspi.yaspi import Yaspi

os.environ['KMP_WARNINGS'] = '0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Execute linear probe experiment')
parser.add_argument('-d', '--datadir', metavar='DIR', default="data", type=Path,
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help=f'model arch: {"|".join(model_names)} (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--classes', default=10, type=int,
                    help='Number of classes in the training set (default:10)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--sigtemp', default=1.0, type=float,
                    help='Pre-quantum Sigmoid temperature (default: 1.0)')

parser.add_argument('--batchnorm', dest='batchnorm', action='store_true',
                    help='If enabled, apply BatchNorm1d to the input of the pre-quantum Sigmoid.')

parser.add_argument('--identity', dest='identity', action='store_true',
                    help='If enabled, the test network is replaced by the identity. The previous and subsequent layer '
                         'still compress to n_qubits however.')
parser.add_argument('-w', '--width', type=int, default=4,
                    help='Width of the test network (default: 4). If quantum, this is the number of qubits.')
parser.add_argument('--layers', type=int, default=2,
                    help='Number of layers in the test network (default: 2).')

parser.add_argument('-q', '--quantum', dest='quantum', action='store_true',
                    help='If enabled, use a minimised version of ResNet-18 with QNet as the final layer')
parser.add_argument('--q_backend', type=str, default='qasm_simulator',
                    help='Type of backend simulator to run quantum circuits on (default: qasm_simulator)')

parser.add_argument('--encoding', type=str, default='vector',
                    help='Data encoding method (default: vector)')
parser.add_argument('--q_ansatz', type=str, default='sim_circ_14_half',
                    help='Variational ansatz method (default: sim_circ_14_half)')
parser.add_argument('--q_sweeps', type=int, default=1,
                    help='Number of ansatz sweeeps.')
parser.add_argument('--activation', type=str, default='partial_measurement_half',
                    help='Quantum layer activation function type (default: partial_measurement_half)')
parser.add_argument('--shots', type=int, default=100,
                    help='Number of shots for quantum circuit evaulations.')
parser.add_argument('--save-dhs', action='store_true',
                    help='If enabled, compute the Hilbert-Schmidt distance of the quantum statevectors belonging to'
                         ' each class. Only works for -q and --classes 2.')

parser.add_argument('--submission-time', type=str, default='',
                    help='Date and time of yaspify submission to create output directory.')

# --------------------------------------------------------------------------------
# cluster grid options
parser.add_argument("--yaspify", action="store_true")
parser.add_argument("--slurm", action="store_true")
parser.add_argument("--worker_id", type=int, default=0)
parser.add_argument("--yaspi_defaults_path", default="yaspi_probe_defaults.json")
parser.add_argument("--exp_config", default="yaspi_probe.json", type=Path)
# --------------------------------------------------------------------------------


best_acc1 = 0
acc1_list = []


def main():
    args = parser.parse_args()

    # --------------------------------------------------------------------------------
    # Support cluster grid search
    if args.yaspify:
        # Load cluster job options
        with open(args.yaspi_defaults_path, "r") as f:
            yaspi_defaults = json.load(f)
        # Load experiment hyperparams for the demo
        with open(args.exp_config, "r") as f:
            exp_kwargs = json.load(f)
        cmd_args = sys.argv
        cmd_args.remove("--yaspify")

        directory = args.pretrained
        cmd_args.remove("--pretrained")
        cmd_args.remove(directory)

        checkpoints = []
        for model in os.listdir(directory):
            for tepoch in exp_kwargs["tepochs"]:
                if type(tepoch) == str and '_' in tepoch:
                    epoch_string, batch_string = tepoch.split('_')
                    fname = f"checkpoint_{int(epoch_string):04d}_{int(batch_string):04d}.path.tar"
                else:
                    fname = f"checkpoint_{int(tepoch):04d}.path.tar"

                checkpoint = os.path.join(directory, model, fname)
                print(checkpoint)
                if os.path.exists(checkpoint):
                    checkpoints.append(checkpoint)
                else:
                    print(f"Cannot find checkpoint {checkpoint}.")
        print(checkpoints)
        del exp_kwargs["tepochs"]
        exp_kwargs["pretrained"] = checkpoints

        cmd_args.append('--submission-time')
        cmd_args.append(datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S"))
        base_cmd = f"python {' '.join(cmd_args)} --slurm"
        job_name = f"train-simclr-{args.exp_config.stem}"
        # compute cartesian product of options
        job_queue = []
        hparam_vals = [x for x in exp_kwargs.values()]
        grid_vals = list(itertools.product(*hparam_vals))
        hparams = list(exp_kwargs)

        for vals in grid_vals:
            kwargs = " ".join(f"--{hparam} {val}" for hparam, val in zip(hparams, vals))
            job_queue.append(f'"{kwargs}"')
        job = Yaspi(
            cmd=base_cmd,
            job_queue=" ".join(job_queue),
            job_name=job_name,
            job_array_size=len(job_queue),
            **yaspi_defaults,
        )
        job.submit(watch=True, conserve_resources=5)
    else:
        if args.slurm:
            # add any cluster-specific setup you need to do here. E.g. I run a script
            # sets up some temporary symlinks
            if getpass.getuser() == "albanie":
                os.system(str(Path.home() / "configure_tmp_data.sh"))

        print('=' * 30)
        print('==> Training Setting: \n {}'.format(args))
        print('=' * 30)

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        if args.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')

        ngpus_per_node = torch.cuda.device_count()

        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global acc1_list
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))

            # Load hyperparams of trained network
            train_args = json.load(open(os.path.join(os.path.dirname(args.pretrained), "train_args.json"), "r"))
            args.arch = train_args["arch"]
            args.identity = train_args["identity"]
            args.width = train_args["width"]
            args.layers = train_args["layers"]
            args.quantum = train_args["quantum"]
            args.q_ansatz = train_args["q_ansatz"]
            args.q_sweeps = train_args["q_sweeps"]
            args.activation = train_args["activation"]
            args.shots = train_args["shots"]

            # create model
            print("=> creating model '{}'".format(args.arch))
            model = moco.builder.SimCLR(hybrid_resnet.resnet18, args=args).encoder

            # freeze all layers but the last fc
            for name, param in model.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False

            # init the fc layer
            model.fc = torch.nn.Linear(model.fc[0].in_features, args.classes, bias=True)

            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('encoder') and not k.startswith('encoder.fc'):
                    # remove prefix
                    state_dict[k[len("encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    else:
        # create model
        print("=> creating model '{}'".format(args.arch))
        model = moco.builder.SimCLR(hybrid_resnet.resnet18, args=args).encoder

        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

        # init the fc layer
        model.fc = torch.nn.Linear(model.fc[0].in_features, args.classes, bias=True)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print('=> Training with CPU.')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-6)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location="cpu")
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    # For CIFAR
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    augmentation = [
        transforms.RandomResizedCrop(32),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
    ]

    num_classes = args.classes

    train_dataset = datasets.CIFAR10(root=args.datadir, train=True, download=True,
                                     transform=transforms.Compose(augmentation))

    train_labels = np.array(train_dataset.targets)
    train_idx = np.array(
        [np.where(train_labels == i)[0] for i in range(0, num_classes)]).flatten()
    train_dataset.targets = train_labels[train_idx]
    train_dataset.data = train_dataset.data[train_idx]

    val_dataset = datasets.CIFAR10(root=args.datadir, train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    val_labels = np.array(val_dataset.targets)
    val_idx = np.array(
        [np.where(val_labels == i)[0] for i in range(0, num_classes)]).flatten()
    val_dataset.targets = val_labels[val_idx]
    val_dataset.data = val_dataset.data[val_idx]

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    model_path = create_output_model_path(args)
    print('Linear probing model saved at {}'.format(model_path))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        # add acc1 to list for printing
        acc1_list.append(round(acc1.item(), 3))
        print(acc1_list)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, model_path)
        if epoch == args.start_epoch and args.pretrained != '':
            sanity_check(model.state_dict(), args.pretrained)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top2],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top2.update(acc2[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top2],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top2.update(acc2[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f}'
              .format(top1=top1, top2=top2))

    return top1.avg


def create_output_model_path(args, version=0):
    if os.path.exists(args.pretrained):
        file_name = f'{args.pretrained.split(os.sep)[-2]}_{args.pretrained.split(os.sep)[-1].split(".")[0]}_{version}'
    else:
        if args.quantum:
            file_name = 'SimCLR-{}-quantum_{}-classes_{}-netwidth_{}-nlayers_{}-nsweeps_{}-activation_{}-shots_{}-bsize_{}-scratch_{}'.format(
                args.arch, args.quantum, args.classes, args.width, args.layers, args.q_sweeps,
                args.activation, args.shots, args.batch_size, version)
        else:
            file_name = 'SimCLR-{}-quantum_{}-classes_{}-netwidth_{}-nlayers_{}-identity_{}-bsize_{}-scratch_{}'.format(
                args.arch, args.quantum, args.classes, args.width, args.layers, args.identity,
                args.batch_size, version)

    model_path = os.path.join('model', 'sup', 'simclr', args.submission_time, file_name)
    if os.path.exists(model_path):
        return create_output_model_path(args, version=version + 1)
    else:
        os.makedirs(model_path)
        return model_path


def save_checkpoint(state, is_best, model_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(model_path, filename))
    if is_best:
        fname = os.path.join(model_path, 'model_best.pth.tar')
        shutil.copyfile(os.path.join(model_path, filename), fname)


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
