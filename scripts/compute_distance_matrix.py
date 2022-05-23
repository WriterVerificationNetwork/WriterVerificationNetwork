import csv
import os
import random

import pandas as pd
import plotly.express as px
import torch
import wandb
from sklearn.manifold import TSNE
from tqdm import tqdm

from dataset.data_loader import WriterDataLoader
from dataset.tm_dataset import TMDataset
from model.model_factory import ModelsFactory
from options.test_options import TestOptions
from utils.transform import get_transforms

args = TestOptions().parse()

colour_map = [ "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",

        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]


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
        dataset_val = TMDataset(args.gt_dir, args.gt_binarized_dir, args.filter_file, transforms, split_from=0,
                                split_to=1, unfold=False, min_n_sample_per_letter=args.min_n_sample_per_letter,
                                min_n_sample_per_class=args.min_n_sample_per_class, letters=args.letters)
        self._model.init_losses('Val', use_weighted_loss=False, dataset=dataset_val, letters=args.letters)
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

    def compute_tms_distances(self, embeddings, n_testing_items=7):
        all_tms = list(embeddings.keys())
        distance_func = torch.nn.MSELoss()
        results = {}
        for i in range(len(all_tms)):
            source_tm = all_tms[i]
            results[source_tm] = {}
            for j in range(len(all_tms)):
                target_tm = all_tms[j]
                distances = []
                for _ in range(10 * n_testing_items):
                    source_features = torch.stack(random.sample(embeddings[source_tm], n_testing_items))
                    target_features = torch.stack(random.sample(embeddings[target_tm], n_testing_items))
                    distances.append(distance_func(source_features, target_features))

                avg_distance = sum(distances) / len(distances)
                results[source_tm][target_tm] = avg_distance.item()
        return results

    def _validate(self):
        # set model to eval
        self._model.set_eval()
        data_loader = self.data_loader_val.get_dataloader()
        embeddings = {}
        for train_batch in tqdm(data_loader):
            input_data = self.__get_data(train_batch, 'img_anchor', 'bin_anchor', 'symbol')
            anchor_out, _ = self._model.compute_loss(input_data, criterion_mode='Val')
            footprints = anchor_out['footprint'].detach().cpu()
            for i, symbol in enumerate(train_batch['symbol']):
                tm_anchor = train_batch['tm_anchor'][i]
                if tm_anchor not in embeddings:
                    embeddings[tm_anchor] = []
                embeddings[tm_anchor].append(footprints[i])

        distance_matrix = self.compute_tms_distances(embeddings)

        lines = []
        for source in distance_matrix:
            line = {}
            for target in distance_matrix[source]:
                line[target] = distance_matrix[source][target]
            lines.append(line)

        vis_dir = os.path.join(args.vis_dir, args.name)
        matrix_file = os.path.join(vis_dir, 'distance_matrix_TM.csv')
        with open(matrix_file, 'w') as f:
            writer = csv.DictWriter(f, distance_matrix.keys())
            writer.writeheader()
            writer.writerows(lines)


if __name__ == "__main__":
    trainer = Trainer()
