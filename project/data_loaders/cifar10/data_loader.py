# Cifar 10 data loader lightning class based on torchvision cifar10 dataset

from argparse import Namespace
import os
import torch as t
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms
import ipdb

from torch.utils.data import DataLoader


class CustomDataModule(pl.LightningDataModule):

    def __init__(self, params: Namespace):
        super().__init__()

        self.data_dir = params.data.location
        self.batch_size = params.data.train_batch_size
        # ipdb.set_trace()

        # transform pil image to tensor
        transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Resize((256, 256))
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ]

        )
        train_trainsform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]
        )
        # self.transform = t.jit.script(self.transform)
        # ipdb.set_trace()
        self.train_set = CIFAR10(
            root=self.data_dir, train=True, download=True, transform=train_trainsform)
        self.test_set = CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform)

        self.num_workers = os.cpu_count()

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
