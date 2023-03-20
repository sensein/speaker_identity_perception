"""Training Learner definitions
"""
import torch
from torch import nn
import pytorch_lightning as pl

from aggregator import Aggregator

class PrecomputedNorm(nn.Module):
    """Normalization using Pre-computed Mean/Std.
    Args:
        stats: Precomputed (mean, std).
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, stats, axis=[1, 2]):
        super().__init__()
        self.axis = axis
        self.mean, self.std = stats

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return ((X - self.mean) / self.std)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(mean={self.mean}, std={self.std}, axis={self.axis})'
        return format_string

class NormalizeBatch(nn.Module):
    """Normalization of Input Batch.

    Note:
        Unlike other blocks, use this with *batch inputs*.

    Args:
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _mean = X.mean(dim=self.axis, keepdims=True)
        _std = torch.clamp(X.std(dim=self.axis, keepdims=True), torch.finfo().eps, torch.finfo().max)
        return ((X - _mean) / _std)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(axis={self.axis})'
        return format_string


class Learner(pl.LightningModule):
    """BYOL-A learner. Shows batch statistics for each epochs."""

    def __init__(self, decoder, config):
        super().__init__()
        self.config = config
        self.lr = config.lr
        self.optimizer = config.optim
        self.decoder = decoder
        self.loss = nn.BCELoss()
        self.aggregator = Aggregator(config.aggregation)
        self.sigmoid = nn.Sigmoid()
        self.norm = NormalizeBatch()
        # self.normalizer = PrecomputedNorm([2.1810803413391113,2.7335219383239746])

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
        
        all_pred = self.sigmoid(self.decoder(self.norm(all_x.squeeze(1))))
        loss = self.loss(all_pred, all_y)
        acc = (torch.round(all_pred) == all_y).float().mean()
        return loss.mean(), acc*100

    def training_step(self, triplet_inputs):
        loss, acc = self._shared_eval_step(triplet_inputs)
        self.log('train_loss', float(loss.mean()), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_acc', float(acc), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, triplet_inputs, _):
        loss, acc = self._shared_eval_step(triplet_inputs)
        self.log('val_loss', float(loss), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc', float(acc), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, inputs, _):
        if self.config.test_data == 'stimuli':
            pair_embeddings = self.aggregator(inputs[0], inputs[1])
            labels = inputs[2].unsqueeze(1)
            predictions = self.sigmoid(self.decoder(self.norm(pair_embeddings.squeeze(1))))
            loss = self.loss(predictions, labels)
            acc = (torch.round(predictions) == labels).float().mean()*100
        else:
            loss, acc = self._shared_eval_step(inputs)
        self.log('test_loss', float(loss), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_acc', float(acc), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            assert False, f'Unknown optimizer: "{self.optimizer}"'
        return optimizer