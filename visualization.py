import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib import cm
from sklearn.manifold import TSNE

from dataset.data_loader import WriterDataLoader
from dataset.image_dataset import ImageDataset
from model.model_factory import ModelsFactory
from options.test_options import TestOptions
from utils.transform import get_transforms

args = TestOptions().parse()


class Trainer:
    def __init__(self):
        wandb.init(group=args.group,
                   project=args.wb_project,
                   entity=args.wb_entity)
        wandb.run.name = args.name
        wandb.run.save()
        wandb.config.update(args)
        device = torch.device('cuda' if args.cuda else 'cpu')
        self._model = ModelsFactory.get_model(args, is_train=True, device=device, dropout=0.5)
        self._model.load_network(args.pretrained_model_path)
        transforms = get_transforms(args)
        dataset_val = ImageDataset(args.gt_dir, args.gt_binarized_dir, args.filter_file, transforms,
                                   split_from=0.8, split_to=1, unfold=False)
        self._model.init_losses('Val', use_weighted_loss=False, dataset=dataset_val)
        self.data_loader_val = WriterDataLoader(dataset_val, is_train=False, numb_threads=args.n_threads_train,
                                                batch_size=args.batch_size)

        print("Validating sets: {} images".format(len(dataset_val)))

        self._validate()

    def __get_data(self, batch_data, image_key, reconstruct_key, symbol_key):
        res = {'image': batch_data[image_key]}
        if 'reconstruct' in args.tasks:
            res['reconstruct'] = batch_data[reconstruct_key] * args.bin_weight
        if 'symbol' in args.tasks:
            res['symbol'] = batch_data[symbol_key]
        return res

    def _validate(self):
        # set model to eval
        self._model.set_eval()
        data_loader = self.data_loader_val.get_dataloader()
        embeddings = {}
        for i_train_batch, train_batch in enumerate(data_loader):
            input_data = self.__get_data(train_batch, 'img_anchor', 'bin_anchor', 'symbol')
            anchor_out, _ = self._model.compute_loss(input_data, criterion_mode='Val')
            for i, symbol in enumerate(train_batch['symbol']):
                symbol = symbol.item()
                footprints = anchor_out['footprint'].detach().cpu()
                if symbol not in embeddings:
                    embeddings[symbol] = {}
                tm_anchor = train_batch['tm_anchor'][i]
                if tm_anchor not in embeddings[symbol]:
                    embeddings[symbol][tm_anchor] = []
                embeddings[symbol][tm_anchor].append(footprints[i])

        for symbol in embeddings:
            tms = list(embeddings[symbol].keys())
            tm_to_idx = {x: i for i, x in enumerate(tms)}
            sym_embedding, tm_tensors = [], []
            for tm in embeddings[symbol]:
                tm_tensors += [tm_to_idx[tm] for _ in range(len(embeddings[symbol][tm]))]
                sym_embedding += embeddings[symbol][tm]

            footprints = torch.stack(sym_embedding, dim=0)
            tm_tensors = torch.tensor(tm_tensors)
            tsne = TSNE(2, verbose=1)
            tsne_proj = tsne.fit_transform(footprints)
            # Plot those points as a scatter plot and label them based on the pred labels
            cmap = cm.get_cmap('tab20')
            fig, ax = plt.subplots(figsize=(8, 8))
            for tm, idx in tm_to_idx.items():
                indices = tm_tensors == idx
                ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap(idx)).reshape(1, 4),
                           label=tm, alpha=0.5)
            # ax.legend(fontsize='large', markerscale=2)
            plt.show()


if __name__ == "__main__":
    trainer = Trainer()
