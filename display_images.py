import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torchvision import datasets

import moco.loader

# which class(es) to add to trainloader (just uses 100 of each)
classes = [0]
# which number image to display
image_id = 0
# bool if should be saved as well as displayed
save = False

# WITH AUGMENTATION
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])

augmentation = [
    transforms.RandomResizedCrop(96, scale=(0.2, 1.0)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    # normalize
]

train_dataset = datasets.STL10(root='./data', download=True,
                               transform=moco.loader.TwoCropsTransform(
                                   transforms.Compose(augmentation)))

train_labels = np.array(train_dataset.labels)
train_idx = np.array(
    [np.where(train_labels == i)[0][:100] for i in classes]).flatten()
train_dataset.labels = train_labels[train_idx]
train_dataset.data = train_dataset.data[train_idx]

train_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=False)
images = [images for batch_index, (images, _) in enumerate(train_loader)]

plt.imshow(torch.tensor(images[image_id][0].squeeze()).permute(1, 2, 0))
plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
plt.margins(0, 0)
if save:
    plt.savefig("image_aug1.pdf", bbox_inches='tight')
plt.show()

plt.imshow(torch.tensor(images[image_id][1].squeeze()).permute(1, 2, 0))
plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
plt.margins(0, 0)
if save:
    plt.savefig("image_aug2.pdf", bbox_inches='tight')
plt.show()

# WITHOUT AUGMENTATION
augmentation = [
    transforms.ToTensor(),
    # normalize
]

train_dataset = datasets.STL10(root='./data', download=True,
                               transform=moco.loader.TwoCropsTransform(
                                   transforms.Compose(augmentation)))

train_labels = np.array(train_dataset.labels)
train_idx = np.array(
    [np.where(train_labels == i)[0][:100] for i in classes]).flatten()
train_dataset.labels = train_labels[train_idx]
train_dataset.data = train_dataset.data[train_idx]

train_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=False)
images = [images for batch_index, (images, _) in enumerate(train_loader)]

plt.imshow(torch.tensor(images[image_id][0].squeeze()).permute(1, 2, 0))
plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
plt.margins(0, 0)
if save:
    plt.savefig("image.pdf", bbox_inches='tight')
plt.show()
