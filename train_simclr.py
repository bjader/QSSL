#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import getpass
import itertools
import json
import logging
import os
import random
import shutil
# Imports for yaspi
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchsummary
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from numpy.linalg import matrix_power

import hybrid_resnet
import moco.builder
import moco.loader
from yaspi.yaspi import Yaspi

os.environ['KMP_WARNINGS'] = '0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Quantum self-sup training')
parser.add_argument('-d', '--datadir', metavar='DIR', default='./data', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model arch: {"|".join(model_names)} (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=50000, type=int,
                    help='size of training set to use (default:50000, size of CIFAR10)')
parser.add_argument('--classes', default=10, type=int,
                    help='Number of classes in the training set (default:10)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[320, 360], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='Save trained model every x epochs or batches (see --save-batches)')
parser.add_argument('--save-batches', dest='save_batches', action='store_true',
                    help='Save model every x batches rather than epochs')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--fdim', default=128, type=int,
                    help='feature dimension (default: 128)')

parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--sigtemp', default=1.0, type=float,
                    help='Pre-quantum Sigmoid temperature (default: 1.0)')

parser.add_argument('--batchnorm', dest='batchnorm', action='store_true',
                    help='If enabled, apply BatchNorm1d to the input of the pre-quantum Sigmoid.')

parser.add_argument('--identity', dest='identity', action='store_true',
                    help='If enabled, the test network is replaced by the identity. The previous and subsequent layer '
                         'still compress to n_qubits however.')
parser.add_argument('-w', '--width', type=int, default=8,
                    help='Width of the test network (default: 8). If quantum, this is the number of qubits.')
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
parser.add_argument('--activation', type=str, default='null',
                    help='Quantum layer activation function type (default: null)')
parser.add_argument('--shots', type=int, default=100,
                    help='Number of shots for quantum circuit evaluations.')
parser.add_argument('--save-dhs', action='store_true',
                    help='If enabled, compute the Hilbert-Schmidt distance of the quantum statevectors belonging to'
                         ' each class. Only works for -q and --classes 2.')
parser.add_argument('--save-overlap', action='store_true',
                    help='If enabled, compute the overlap between statevectors corresponding to positive training'
                         'pairs. Saves average for overlap of each batch.')

parser.add_argument('--submission-time', type=str, default='',
                    help='Date and time of yaspify submission to create output directory.')

# --------------------------------------------------------------------------------
# cluster grid optionsssh
parser.add_argument("--yaspify", action="store_true")
parser.add_argument("--slurm", action="store_true")
parser.add_argument("--worker_id", type=int, default=0)
parser.add_argument("--yaspi_defaults_path", default="yaspi_train_defaults.json")
parser.add_argument("--exp_config", default="yaspi_train.json", type=Path)


# --------------------------------------------------------------------------------

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
    args.gpu = gpu
    if args.gpu is not None:
        print('=' * 30)
        print("Use GPU: {} for training".format(args.gpu))
        print('=' * 30)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.SimCLR(hybrid_resnet.resnet18, args.width, args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print('=> Training with CPU.')

    torchsummary.summary(model, (3, 32, 32))

    # define loss function (criterion) and optimizer
    criterion = None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Normalization for ImageNet
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    # Normalization for CIFAR
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    augmentation = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.CIFAR10(root=args.datadir, train=True, download=True,
                                     transform=moco.loader.TwoCropsTransform(
                                         transforms.Compose(augmentation)))

    train_labels = np.array(train_dataset.targets)
    num_classes = args.classes
    train_idx = np.array(
        [np.where(train_labels == i)[0][:int(args.epoch_size / num_classes)] for i in range(0, num_classes)]).flatten()
    train_dataset.targets = train_labels[train_idx]
    train_dataset.data = train_dataset.data[train_idx]

    if len(train_idx) < args.epoch_size:
        logging.warning(
            f"Requested epoch size ({args.epoch_size}) is greater than available images for chosen classes "
            f"({len(train_idx)}). Training will use the set of available images, output files will save requested "
            f"epoch size.")

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    model_path = create_output_model_path(args)
    print('Training model saved at {}'.format(model_path))

    with open(os.path.join(model_path, "train_args.json"), 'w') as fp:
        args.exp_config = str(args.exp_config)
        json.dump(vars(args), fp)

    repr_network_params = []
    dhs_list = []
    dhs_positive_pair_list = []
    overlap_list = []
    loss_list = []

    # Wipe metric information accumulated during TorchSummary
    if hasattr(model.encoder.repr_network[0], 'qnn'):
        model.encoder.repr_network[0].qnn.gradients = []
        model.encoder.repr_network[0].qnn.statevectors = []

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, model_path, criterion, optimizer, epoch, args, repr_network_params, dhs_list,
              dhs_positive_pair_list,
              overlap_list, loss_list)

        if not args.save_batches:
            fname = 'checkpoint_{:04d}.path.tar'.format(epoch)
            checkpoint_name = os.path.join(model_path, fname)
            if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
                ckpt = {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(ckpt, is_best=False, filename=checkpoint_name)


def train(train_loader, model, model_path, criterion, optimizer, epoch, args, repr_network_params, dhs_list,
          dhs_positive_pair_list, overlap_list, loss_list):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top2],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    # switch to train mode
    model.train()

    end = time.time()

    for batch_index, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Wipe the previous quantum statevectors
        if hasattr(model.encoder.repr_network[0], 'qnn'):
            model.encoder.repr_network[0].qnn.statevectors = []

        # Labels are NOT used for training, only for quantum metrics
        labels = _

        target = torch.zeros(2 * args.batch_size, dtype=torch.long)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
        out_1 = model(x=images[0])
        out_2 = model(x=images[1])

        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)

        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=sim_matrix.device)).type(torch.bool)
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        loss_list.append(loss.item())

        # to compute acc, concate the positive and negatives
        M = torch.cat([pos_sim.view(2 * args.batch_size, 1), sim_matrix], dim=-1)
        acc1, acc2 = accuracy(M, target, topk=(1, 2))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], 2 * args.batch_size)
        top2.update(acc2[0], 2 * args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % args.print_freq == 0:
            progress.display(batch_index)

        if args.save_batches:
            if batch_index % args.save_freq == 0 or batch_index == np.floor(args.epoch_size / args.batch_size):
                fname = 'checkpoint_{:04d}_{:04d}.path.tar'.format(epoch, batch_index)
                checkpoint_name = os.path.join(model_path, fname)

                ckpt = {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(ckpt, is_best=False, filename=checkpoint_name)

                if not args.identity:
                    parameters = list(model.encoder.repr_network[0].parameters())
                    repr_network_params.append(parameters[0].detach().cpu().numpy().flatten().tolist())

                metrics_name = os.path.join(model_path, 'training_metrics')

                if hasattr(model.encoder.repr_network[0], 'qnn'):
                    gradients = model.encoder.repr_network[0].qnn.gradients
                    if args.q_backend == 'statevector_simulator':
                        statevectors = np.array(model.encoder.repr_network[0].qnn.statevectors)
                        if args.save_dhs:
                            labels = np.array(labels)

                            total_labels = np.append(labels, labels)

                            class_0_statevectors = statevectors[total_labels == 0]
                            class_1_statevectors = statevectors[total_labels != 0]

                            rho = np.mean([np.outer(vector, np.conj(vector)) for vector in class_0_statevectors],
                                          axis=0)
                            sigma = np.mean([np.outer(vector, np.conj(vector)) for vector in class_1_statevectors],
                                            axis=0)

                            rho_squared = np.trace(matrix_power(rho, 2))
                            sigma_squared = np.trace(matrix_power(sigma, 2))
                            rho_sigma = np.trace(np.matmul(rho, sigma))
                            dhs = np.trace(matrix_power((rho - sigma), 2))
                            dhs_list.append([rho_squared.real, sigma_squared.real, rho_sigma.real, dhs.real])

                            # Calculate DHS for positive pair as one class
                            aug_1_statevectors = statevectors[:int(len(statevectors) / 2)]
                            aug_2_statevectors = statevectors[int(len(statevectors) / 2):]

                            positive_pairs = list(zip(aug_1_statevectors, aug_2_statevectors))
                            rhos = []
                            sigmas = []

                            for i, positive_pair in enumerate(positive_pairs):
                                rho = np.mean([np.outer(vector, np.conj(vector)) for vector in positive_pair], axis=0)
                                rhos.append(rho)

                                negatives = positive_pairs[:i] + positive_pairs[i + 1:]
                                sigma = np.mean(
                                    [np.outer(vector, np.conj(vector)) for pair in negatives for vector in pair],
                                    axis=0)
                                sigmas.append(sigma)

                            average_rho_squared = np.mean([np.trace(matrix_power(rho, 2)) for rho in rhos], axis=0)
                            average_sigma_squared = np.mean([np.trace(matrix_power(sigma, 2)) for sigma in sigmas],
                                                            axis=0)
                            average_rho_sigma = np.mean(
                                [np.trace(np.matmul(rho, sigma)) for (rho, sigma) in zip(rhos, sigmas)], axis=0)

                            average_dhs = np.mean(
                                [np.trace(matrix_power((rho - sigma), 2)) for (rho, sigma) in zip(rhos, sigmas)],
                                axis=0)

                            dhs_positive_pair_list.append(
                                [average_rho_squared.real, average_sigma_squared.real, average_rho_sigma.real,
                                 average_dhs.real])

                        if args.save_overlap:
                            aug_1_statevectors = statevectors[:int(len(statevectors) / 2)]
                            aug_2_statevectors = statevectors[int(len(statevectors) / 2):]
                            positive_pairs_overlaps = [abs(np.dot(np.conj(vec_1), vec_2)) ** 2 for (vec_1, vec_2)
                                                       in zip(aug_1_statevectors, aug_2_statevectors)]
                            overlap_list.append(np.mean(positive_pairs_overlaps))

                else:
                    gradients = []
                np.save(metrics_name, np.array([loss_list, repr_network_params, gradients, dhs_list,
                                                dhs_positive_pair_list, overlap_list], dtype=object))


def create_output_model_path(args, version=0):
    if args.quantum:
        model_path = os.path.join('model', 'selfsup', 'simclr', args.submission_time,
                                  'SimCLR-{}-quantum_{}-backend_{}-classes_{}--ansatz_{}-netwidth_{}-nlayers_{}'
                                  '-nsweeps_{}-activation_{}-shots_{}-epochsize_{}-bsize_{}-tepochs_{}_{}'.format(
                                      args.arch, args.quantum, args.q_backend, args.classes, args.q_ansatz, args.width,
                                      args.layers, args.q_sweeps, args.activation, args.shots, args.epoch_size,
                                      args.batch_size, args.epochs, version))
    else:
        model_path = os.path.join('model', 'selfsup', 'simclr', args.submission_time,
                                  'SimCLR-{}-quantum_{}-classes_{}-netwidth_{}-nlayers_{}-identity_{}-epochsize_{}-'
                                  'bsize_{}-tepochs_{}_{}'.format(
                                      args.arch, args.quantum, args.classes, args.width, args.layers, args.identity,
                                      args.epoch_size, args.batch_size, args.epochs, version))

    if os.path.exists(model_path):
        return create_output_model_path(args, version=version + 1)
    else:
        os.makedirs(model_path)
        return model_path


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
        logging.info('\t'.join(entries))

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


def accuracy(output, target, topk=(1, 2)):
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
