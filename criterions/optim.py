from typing import Tuple

import torch
import torch.nn as nn

__all__ = ['Optimizer', 'Scheduler', 'Criterion']

from dataset.utils import rename


class Optimizer:
    def __init__(self):
        pass
    
    def get(self, model: nn.Module, optimizer: str, lr: float, wd: float = 0., momentum: float = 0.5,
            betas: Tuple[float, float] = (0.9, 0.999)):

        if optimizer.lower() == 'sgd':
            optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        elif optimizer.lower() == 'adam':
            optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
        elif optimizer.lower() == 'none':
            optim = Optimizer()
        else:
            raise ValueError("Optimizer {} not supported".format(optimizer))
        return optim


class Scheduler:
    def __init__(self):
        pass
    
    def get( self, lr_scheduler: str, optimizer: torch.optim.Optimizer, step_size: int, gamma: float = 0.1):
        if lr_scheduler.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
        elif lr_scheduler.lower() == 'none':
            scheduler = Scheduler()
        else:
            raise ValueError("lr_scheduler {} not supported".format(lr_scheduler))
        return scheduler
    

class Criterion:
    def __init__(self, eps=1e-8):
        self.eps = eps
    
    def get(self, loss: str, reduction: str = 'mean', weight=None, pos_weight=None):
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.loss_name = loss
        loss_name = loss
        if '+' in loss_name:
            loss_name = loss_name.split('+')
        else:
            loss_name = [loss_name]
        loss_funcs = []
        for single_loss in loss_name:
            loss_func = self.get_single_loss(single_loss)
            loss_funcs.append(loss_func)

        @rename(self.loss_name)
        def combine_losses(*args, **kwargs):
            _loss = 0
            for _loss_func in loss_funcs:
                _loss += _loss_func(*args, **kwargs)
            return _loss
        return combine_losses

    def get_single_loss(self, loss):
        if loss.lower() == 'ce':
            loss_func = nn.CrossEntropyLoss(reduction=self.reduction, weight=self.weight)
        elif loss.lower() == 'bce':
            loss_func = nn.BCEWithLogitsLoss(reduction=self.reduction, weight=self.weight, pos_weight=self.pos_weight)
        elif loss.lower() == 'mse':
            loss_func = nn.MSELoss(reduction=self.reduction)
        elif loss.lower() == 'mae':
            loss_func = nn.MSELoss(reduction=self.reduction)
        elif loss.lower() == 'l1' or loss.lower() == 'l1_loss':
            loss_func = nn.L1Loss(reduction=self.reduction)
        elif loss.lower() == 'triplet':
            loss_func = nn.TripletMarginLoss(reduction=self.reduction)
        else:
            raise NotImplementedError(f'Loss {loss} hasn\'t implemented yet!!!')

        @rename(loss)
        def inner_func(*args, **kwargs):
            return loss_func(*args, **kwargs)
        return inner_func



