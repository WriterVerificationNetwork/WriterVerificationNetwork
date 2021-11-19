import os
import statistics

import numpy as np
import torch

from criterions.optim import Optimizer, Scheduler, Criterion
from utils.misc import map_location


class ModelWrapper:
    def __init__(self, args, model, is_train, device):
        self._name = args.name
        self._model = model
        self._args = args
        self._is_train = is_train
        self._save_dir = os.path.join(self._args.checkpoints_dir, self.name)

        if self._is_train:
            self._model.train()
        else:
            self._model.eval()

        self._init_train_vars()
        self.loss_dict = {}
        self._device = device
        self._criterions_per_task = {}

    @property
    def name(self):
        return self._name

    @property
    def is_train(self):
        return self._is_train

    def _init_train_vars(self):
        self._optimizer = Optimizer().get(self._model, self._args.optimizer, lr=self._args.lr,
                                          wd=self._args.weight_decay)
        self.lr_scheduler = Scheduler().get(self._args.lr_policy, self._optimizer, step_size=self._args.lr_decay_epochs)

    def load(self):
        # load feature extractor
        self._load_network(self._model, self.name)
        self._load_optimizer(self._optimizer, self.name)

    def get_current_lr(self):
        lr = []
        for param_group in self._optimizer.param_groups:
            lr.append(param_group['lr'])
        print('current learning rate: {}'.format(np.unique(lr)))

    def save(self):
        """
        save network, the filename is specified with the sofar tasks and iteration
        """
        self._save_network(self._model, self.name)
        # save optimizers
        self._save_optimizer(self._optimizer, self.name)

    def _save_optimizer(self, optimizer, optimizer_label):
        save_filename = 'opt_%s.pth' % optimizer_label
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def _load_optimizer(self, optimizer, optimizer_label):
        load_filename = 'opt_%s.pth' % optimizer_label
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(load_path), 'Weights file %s not found!' % load_path

        optimizer.load_state_dict(torch.load(load_path))
        print('loaded optimizer: %s' % load_path)

    def _save_network(self, network, network_label):
        save_filename = 'net_%s.pth' % network_label
        save_path = os.path.join(self._save_dir, save_filename)
        save_dict = network.state_dict()
        torch.save(save_dict, save_path)
        print('saved net: %s' % save_path)

    def _load_network(self, network, network_label):
        load_filename = 'net_%s.pth' % network_label
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(load_path), 'Weights file %s not found ' % load_path
        checkpoint = torch.load(load_path, map_location=map_location(self._args.cuda))
        network.load_state_dict(checkpoint)
        print('loaded net: %s' % load_path)

    def set_current_losses(self, loss_dict):
        for key in loss_dict:
            if key not in self.loss_dict:
                self.loss_dict[key] = []
            self.loss_dict[key].append(loss_dict[key])

    def reset_train_losses(self):
        self.loss_dict = {}

    def get_current_losses(self):
        results = {x: statistics.mean(self.loss_dict[x]) for x in self.loss_dict}
        self.reset_train_losses()
        return results

    def set_train(self):
        self._model.train()
        self._is_train = True

    def init_losses(self, mode, use_weighted_loss, dataset):
        # get the training loss
        losses = {}
        for i, task in enumerate(self._args.tasks):
            weight, pos_weight = None, None
            criterion = Criterion()
            losses[task] = criterion.get(self._args.criterion[i], weight=weight, pos_weight=pos_weight)
        self._criterions_per_task[mode] = losses

    def normalize_lambda(self, task):
        lambda_dict = {k:self._args.lambda_task[i] for i, k in enumerate(self._args.tasks)}
        summation = sum([lambda_dict[key] for key in lambda_dict.keys()])
        return dict([(k, lambda_dict[k]/summation) for k in lambda_dict.keys()])[task]

    def set_eval(self):
        self._model.eval()
        self._is_train = False

    def compute_footprint(self, anchor, positive, negative):
        criterion_task = self._criterions_per_task['Train']['footprint']
        loss_task = criterion_task(anchor, positive, negative)
        return self.normalize_lambda('footprint') * loss_task

    def compute_loss(self, batch_data):
        train_dict = dict()
        loss = torch.tensor(0.).to(self._device)
        input_image = batch_data['image'].to(self._device, non_blocking=True)
        with torch.set_grad_enabled(self._is_train):
            output = self._model(input_image)
        for t in [x for x in self._args.tasks if x != 'footprint']:
            label = batch_data[t].to(self._device, non_blocking=True)
            criterion_task = self._criterions_per_task['Train'][t]
            loss_task = criterion_task(output[t], label)
            train_dict['loss_' + t] = loss_task.item()
            loss += self.normalize_lambda(t) * loss_task
        train_dict['loss'] = loss.item()
        return output, loss, train_dict

    def optimise_params(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
