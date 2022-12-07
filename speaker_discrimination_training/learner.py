"""Training Learner definitions
"""
import torch
from torch import nn
import pytorch_lightning as pl

from aggregator import Aggregator


class Learner(pl.LightningModule):
    """BYOL-A learner. Shows batch statistics for each epochs."""

    def __init__(self, decoder, lr, aggregation):
        super().__init__()
        self.lr = lr
        self.decoder = decoder
        self.loss = nn.BCELoss()
        self.aggregator = Aggregator(aggregation)
        self.sigmoid = nn.Sigmoid()

    def training_step(self, triplet_inputs):
        same_pair = self.aggregator(triplet_inputs[0], triplet_inputs[1]) # (B,E)
        diff_pair = self.aggregator(triplet_inputs[0], triplet_inputs[2]) # (B,E)

        same_pred = self.sigmoid(self.decoder(same_pair.squeeze(1)))
        diff_pred = self.sigmoid(self.decoder(diff_pair.squeeze(1)))

        loss_same = self.loss(same_pred.squeeze(1), torch.zeros(same_pred.shape[0]).to('cuda'))
        loss_diff = self.loss(diff_pred.squeeze(1), torch.ones(same_pred.shape[0]).to('cuda'))

        loss = loss_same + loss_diff
        self.log('Loss', float(loss.mean()), prog_bar=True, on_step=False, on_epoch=True)
        return loss.mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)