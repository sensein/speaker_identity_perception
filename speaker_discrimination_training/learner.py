"""Training Learner definitions
"""
import torch
from torch import nn
import pytorch_lightning as pl

from aggregator import Aggregator


class Learner(pl.LightningModule):
    """BYOL-A learner. Shows batch statistics for each epochs."""

    def __init__(self, decoder, clf, lr, optimizer, aggregation):
        super().__init__()
        self.lr = lr
        self.optimizer = optimizer
        self.decoder = decoder
        self.clf = clf
        self.loss = nn.BCELoss()
        self.aggregator = Aggregator(aggregation)
        self.sigmoid = nn.Sigmoid()

        # send a mock image tensor to instantiate singleton parameters
        # with torch.no_grad():
        #     self.forward(torch.randn(2, channels, image_size[0], image_size[1]),
        #                  torch.randn(2, channels, image_size[0], image_size[1]))
    
    def _shared_eval_step(self, triplet_inputs):
        same_pair = self.aggregator(triplet_inputs[0], triplet_inputs[1]) # (B,E)
        diff_pair = self.aggregator(triplet_inputs[0], triplet_inputs[2]) # (B,E)
        same_actual = torch.zeros((same_pair.shape[0],1)).to('cuda')
        diff_actual = torch.ones((diff_pair.shape[0],1)).to('cuda')

        all_x = torch.cat((same_pair, diff_pair), 0)
        all_y = torch.cat((same_actual, diff_actual), 0)

        indices = torch.randperm(all_x.size()[0])
        all_x=all_x[indices]
        all_y=all_y[indices]
        
        all_pred = self.sigmoid(self.clf(self.decoder(all_x.squeeze(1))))

        loss = self.loss(all_pred, all_y)
        acc = (torch.round(all_pred) == all_y).float().mean()
        return loss, acc*100

    def training_step(self, triplet_inputs):
        loss, acc = self._shared_eval_step(triplet_inputs)
        self.log('train_loss', float(loss.mean()), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_acc', float(acc), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss.mean()
    
    @torch.no_grad()
    def validation_step(self, triplet_inputs, _):
        loss, acc = self._shared_eval_step(triplet_inputs)
        self.log('val_loss', float(loss.mean()), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc', float(acc), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def test_step(self, triplet_inputs, _):
        loss, acc = self._shared_eval_step(triplet_inputs)
        self.log('test_loss', float(loss.mean()), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_acc', float(acc), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            assert False, f'Unknown optimizer: "{self.optimizer}"'
        return optimizer