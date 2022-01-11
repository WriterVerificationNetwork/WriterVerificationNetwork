import math
from collections import Counter

import numpy as np
from numpy import interp
from torch.utils.data import DataLoader, WeightedRandomSampler


class WriterDataLoader:

    def __init__(self, dataset, is_train, numb_threads, batch_size, using_sampler=False):
        self._is_train = is_train
        self.batch_size = batch_size
        self.numb_threads = numb_threads
        self.dataset = dataset
        self.use_sampler = using_sampler

    def get_dataloader(self):
        sampler = None
        if self.use_sampler:
            sampler = WriterDataLoader.get_sampler(self.dataset)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            num_workers=int(self.numb_threads),
            sampler=sampler,
            pin_memory=True,
            drop_last=self._is_train)

    @staticmethod
    def get_sampler(dataset, weight_max=10):
        labels = dataset.anchor_tms
        bincount = dict(Counter(labels))
        key_list = list(bincount.keys())
        weight_map = 1 / np.asarray([bincount[x] for x in key_list])
        weight_map = weight_map / np.min(weight_map) * 10
        weight_map = np.log10(weight_map)
        weight_map = interp(weight_map, [np.min(weight_map), np.max(weight_map)], [1, weight_max])
        weights = []
        for key in labels:
            weights.append(weight_map[key_list.index(key)])
        sampler = WeightedRandomSampler(weights, len(labels), replacement=True)
        return sampler

