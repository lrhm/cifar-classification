from argparse import Namespace
from pytorch_lightning import LightningModule
import torch as t
from torch import nn
import mate
import ipdb


class VitClassificationModule(LightningModule):

    def __init__(self, classifier: nn.Module, params: Namespace, *args):
        super().__init__(*args)

        self.params = params
        self.save_hyperparameters(params)
        self.criterion = t.nn.CrossEntropyLoss()
        # self.classifier: t.nn.Module
        self.classifier = classifier

        self.loss = lambda y_hat, y: self.criterion(y_hat, y)

        # disable automatic optimization
        # self.automatic_optimization = False

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # ipdb.set_trace(   )
        loss = self.loss(y_hat, y)
        # loss = self.distiller(x, y)
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
        # ipdb.set_trace()
        from yerbamate.bunch import Bunch
        # self.params.optimizer
        return mate.Optimizer(Bunch(self.params.optimizer), self.distiller)()
