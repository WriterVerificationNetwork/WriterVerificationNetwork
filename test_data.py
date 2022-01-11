import gc
import time

import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt

from dataset.data_loader import WriterDataLoader
from dataset.tm_dataset import TMDataset
from model.model_factory import ModelsFactory
from options.train_options import TrainOptions
from utils.misc import EarlyStop
from utils.transform import get_transforms
from utils.wb_utils import log_prediction

args = TrainOptions().parse()


class Trainer:
    def __init__(self):
        device = torch.device('cuda' if args.cuda else 'cpu')
        self._model = ModelsFactory.get_model(args, is_train=True, device=device, dropout=0.5)
        transforms = get_transforms(args)
        dataset_train = TMDataset(args.gt_dir, args.gt_binarized_dir, args.filter_file, transforms,
                                  split_from=0, split_to=0.8, min_n_sample_per_letter=args.min_n_sample_per_letter,
                                  min_n_sample_per_class=args.min_n_sample_per_class)
        self._model.init_losses('Train', args.use_weighted_loss, dataset_train)
        self.data_loader_train = WriterDataLoader(dataset_train, is_train=True, numb_threads=args.n_threads_train,
                                                  batch_size=args.batch_size, using_sampler=args.use_sampler)
        self._train()

    def _train(self):
        data_loader = self.data_loader_train.get_dataloader()
        all_categories = {}
        for i in range(10):
            for i_train_batch, train_batch in enumerate(data_loader):
                for tm in train_batch['tm_anchor']:
                    if tm not in all_categories:
                        all_categories[tm] = 0
                    all_categories[tm] += 1

        for tm in all_categories:
            all_categories[tm] /= 10

        print(all_categories)
        df = pd.DataFrame({'TMs': all_categories})
        df.plot(kind='bar',
                  stacked=False,
                  title=f'PSPI - Weight max: {args.pspi_sampler_weight_max}, Drop lv0 rate: {args.lv0_drop_rate}')
        plt.show()



if __name__ == "__main__":
    trainer = Trainer()
