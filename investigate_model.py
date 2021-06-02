from torchvision import models

import moco.builder

model = moco.builder.SimCLR(models.__dict__['resnet18'], 128)
