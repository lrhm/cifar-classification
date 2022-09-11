from argparse import Namespace
from pytorch_lightning import LightningModule
import torch as t
import mate
import ipdb


class Model(LightningModule):

    def __init__(self, params: Namespace):
        super().__init__()

        self.params = params
        self.criterion = t.nn.CrossEntropyLoss()
        self.classifier: t.nn.Module

        self.loss = lambda y_hat, y: self.criterion(y_hat, y)

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # ipdb.set_trace()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return mate.Optimizer(self.params.configure_optimizers, self.classifier)()
