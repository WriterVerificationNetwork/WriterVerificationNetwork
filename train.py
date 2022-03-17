import gc
import random
import time
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from dataset.data_loader import WriterDataLoader
from dataset.tm_dataset import TMDataset
from model.model_factory import ModelsFactory
from options.train_options import TrainOptions
from utils.misc import EarlyStop
from utils.transform import get_transforms
from utils.wb_utils import log_prediction

args = TrainOptions().parse()

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
        transforms = get_transforms(args)
        dataset_train = TMDataset(args.gt_dir, args.gt_binarized_dir, args.filter_file, transforms,
                                  split_from=0, split_to=0.8, min_n_sample_per_letter=args.min_n_sample_per_letter,
                                  min_n_sample_per_class=args.min_n_sample_per_class)
        self._model.init_losses('Train', args.use_weighted_loss, dataset_train)
        self.data_loader_train = WriterDataLoader(dataset_train, is_train=True, numb_threads=args.n_threads_train,
                                                  batch_size=args.batch_size, using_sampler=args.use_sampler)

        dataset_val = TMDataset(args.gt_dir, args.gt_binarized_dir, args.filter_file, transforms, split_from=0.8,
                                split_to=1, unfold=True, min_n_sample_per_letter=args.min_n_sample_per_letter,
                                min_n_sample_per_class=args.min_n_sample_per_class)
        self._model.init_losses('Val', use_weighted_loss=False, dataset=dataset_val)
        self.data_loader_val = WriterDataLoader(dataset_val, is_train=False, numb_threads=args.n_threads_train,
                                                batch_size=args.batch_size)

        self.early_stop = EarlyStop(args.early_stop)
        print("Training tasks {}".format(args.tasks))
        print("Training sets: {} images".format(len(dataset_train)))
        print("Validating sets: {} images".format(len(dataset_val)))

        self._train()
        self._model.load()
        self._visualize(0, 1, "all_data")
        self._visualize(0.8, 1, "val_set")

    def _train(self):
        self._current_step = 0
        self._last_save_time = time.time()
        best_val_loss = 99999
        for i_epoch in range(1, args.nepochs + 1):
            epoch_start_time = time.time()
            self._model.get_current_lr()
            # train epoch
            self._train_epoch(i_epoch)
            if args.lr_policy == 'step':
                self._model.lr_scheduler.step()

            if not i_epoch % args.n_epochs_per_eval == 0:
                continue

            val_dict = self._validate(i_epoch, self.data_loader_val.get_dataloader())
            gc.collect()

            current_loss = val_dict['val/loss_footprint']
            if current_loss < best_val_loss:
                print("Footprint val loss improved, from {:.4f} to {:.4f}".format(best_val_loss, current_loss))
                best_val_loss = current_loss
                for key in val_dict:
                    wandb.run.summary[f'best_model/{key}'] = val_dict[key]
                self._model.save()  # save best model

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, args.nepochs, time_epoch, time_epoch / 60, time_epoch / 3600))

            if self.early_stop.should_stop(current_loss):
                print(f'Early stop at epoch {i_epoch}')
                break

    def __get_data(self, batch_data, image_key, reconstruct_key, symbol_key):
        res = {'image': batch_data[image_key]}
        if 'reconstruct' in args.tasks:
            res['reconstruct'] = batch_data[reconstruct_key] * args.bin_weight
        if 'symbol' in args.tasks:
            res['symbol'] = batch_data[symbol_key]
        return res

    def _compute_loss(self, batch_data, log_data=False, n_log_items=10):
        input_data = self.__get_data(batch_data, 'img_anchor', 'bin_anchor', 'symbol')
        anchor_out, anchor_losses = self._model.compute_loss(input_data)
        input_data = self.__get_data(batch_data, 'img_positive', 'bin_positive', 'symbol')
        pos_out, pos_losses = self._model.compute_loss(input_data)
        input_data = self.__get_data(batch_data, 'img_negative', 'bin_negative', 'symbol')
        neg_out, neg_losses = self._model.compute_loss(input_data)
        footprint_loss = self._model.compute_footprint(anchor_out['footprint'], pos_out['footprint'],
                                                       neg_out['footprint'])
        final_losses = {}
        for task in anchor_losses:
            final_losses[task] = (anchor_losses[task] + pos_losses[task] + neg_losses[task]) / 3.
        final_losses['footprint'] = footprint_loss

        if log_data:
            columns = ['id', 'anchor']
            if 'reconstruct' in args.tasks:
                columns += ['anchor_bin', 'anchor_bin_pred']
            columns += ['positive', 'negative']
            if 'symbol' in args.tasks:
                columns += ['symbol', 'symbol_pred']
            columns += ['pos_distance', 'neg_distance']
            wb_table = wandb.Table(columns=columns)
            log_prediction(wb_table, columns, batch_data, anchor_out, pos_out, neg_out,
                           n_items=n_log_items, bin_weight=args.bin_weight)
            wandb.log({'val_prediction': wb_table}, step=self._current_step)

        accuracies = {}
        if 'symbol' in args.tasks:
            anchor_pred = torch.max(anchor_out['symbol'], dim=1).indices.cpu() == batch_data['symbol']
            pos_pred = torch.max(pos_out['symbol'], dim=1).indices.cpu() == batch_data['symbol']
            neg_pred = torch.max(neg_out['symbol'], dim=1).indices.cpu() == batch_data['symbol']
            accuracies['symbol'] = torch.cat([anchor_pred, pos_pred, neg_pred], dim=0).type(torch.int8).tolist()

        if 'footprint' in args.tasks:
            distance_func = torch.nn.MSELoss(reduction='none')
            anchor_pos_distance = distance_func(anchor_out['footprint'], pos_out['footprint']).mean(dim=1)
            anchor_neg_distance = distance_func(anchor_out['footprint'], neg_out['footprint']).mean(dim=1)
            accuracies['footprint'] = (anchor_neg_distance > anchor_pos_distance).type(torch.int8).tolist()

        log_info = {f'loss_{k}': v.item() for k, v in final_losses.items()}
        final_loss = 0.
        for task in final_losses:
            final_loss += self._model.normalize_lambda(task) * final_losses[task]
        log_info['loss'] = final_loss.item()
        return final_loss, log_info, accuracies

    def _train_epoch(self, i_epoch):
        self._model.set_train()
        data_loader = self.data_loader_train.get_dataloader()
        all_accuracies = {}
        for i_train_batch, train_batch in enumerate(data_loader):
            iter_start_time = time.time()

            # display flags
            do_save = time.time() - self._last_save_time > args.save_freq_s
            final_loss, log_info, accuracies = self._compute_loss(train_batch)
            for key in accuracies:
                if key not in all_accuracies:
                    all_accuracies[key] = []
                all_accuracies[key] += accuracies[key]
            # train model
            self._model.optimise_params(final_loss)
            self._model.set_current_losses(log_info)

            # update epoch info
            self._current_step += 1

            if do_save:
                save_dict = {}
                loss_dict = self._model.get_current_losses()
                for key in loss_dict:
                    save_dict[f'train/{key}'] = loss_dict[key]
                for key in all_accuracies:
                    save_dict[f'train/acc/{key}'] = sum(all_accuracies[key]) / len(all_accuracies[key])
                    all_accuracies[key] = []
                wandb.log(save_dict, step=self._current_step)
                self._display_terminal(iter_start_time, i_epoch, i_train_batch, len(data_loader), loss_dict)
                self._last_save_time = time.time()

    def plot_fig(self, embeddings, tm_tensors, tm_to_idx, tsne_proj, n_item_to_plot):
        fig, ax = plt.subplots(figsize=(12, 12))
        embedding_lens = [{'len': len(v), 'key': k} for k, v in embeddings.items()]
        embedding_lens = sorted(embedding_lens, key=lambda x: x['len'], reverse=True)

        for count, item in enumerate(embedding_lens[:n_item_to_plot]):
            idx = tm_to_idx[item['key']]
            indices = tm_tensors == idx
            ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=colour_map[count],
                       label=item['key'], alpha=0.5)
        ax.legend(fontsize='small', markerscale=2, bbox_to_anchor=(1, 1))
        image_data = BytesIO()  # Create empty in-memory file
        fig.savefig(image_data, format='png', dpi=100)  # Save pyplot figure to in-memory file
        image_data.seek(0)  # Move stream position back to beginning of file
        file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
        plt.close(fig)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    def compute_tms_distances(self, embeddings, n_testing_items=10):
        all_tms = list(embeddings.keys())
        distance_func = torch.nn.MSELoss()
        results = []
        for i in range(len(all_tms)):
            source_tm = all_tms[i]
            for j in range(i + 1, len(all_tms)):
                target_tm = all_tms[j]
                distances = []
                for _ in range(3 * n_testing_items):
                    source_features = torch.stack(random.sample(embeddings[source_tm], n_testing_items))
                    target_features = torch.stack(random.sample(embeddings[target_tm], n_testing_items))
                    distances.append(distance_func(source_features, target_features))

                avg_distance = sum(distances) / len(distances)
                results.append({'source': source_tm, 'target': target_tm, 'distance': avg_distance.item()})
        return results

    def _visualize(self, split_from, split_to, viz_name):
        self._model.set_eval()
        transforms = get_transforms(args)
        dataset_val = TMDataset(args.gt_dir, args.gt_binarized_dir, args.filter_file, transforms,
                                split_from=split_from,
                                split_to=split_to, unfold=False, min_n_sample_per_letter=args.min_n_sample_per_letter,
                                min_n_sample_per_class=args.min_n_sample_per_class)
        data_loader_val = WriterDataLoader(dataset_val, is_train=False, numb_threads=args.n_threads_train,
                                           batch_size=args.batch_size)
        data_loader = data_loader_val.get_dataloader()
        embeddings = {}
        for train_batch in tqdm(data_loader):
            input_data = self.__get_data(train_batch, 'img_anchor', 'bin_anchor', 'symbol')
            anchor_out, _ = self._model.compute_loss(input_data, criterion_mode='Val')
            footprints = anchor_out['footprint'].detach().cpu()
            for i, symbol in enumerate(train_batch['symbol']):
                # symbol = symbol.item()
                # if symbol not in embeddings:
                #     embeddings[symbol] = {}
                tm_anchor = train_batch['tm_anchor'][i]
                if tm_anchor not in embeddings:
                    embeddings[tm_anchor] = []
                embeddings[tm_anchor].append(footprints[i])
        distance_data = self.compute_tms_distances(embeddings)
        distance_data = sorted(distance_data, key=lambda x: x['distance'])
        distance_table = wandb.Table(dataframe=pd.DataFrame(distance_data))
        wandb.log({f'distance_{viz_name}': distance_table}, step=self._current_step)

        tms = list(embeddings.keys())
        tm_to_idx = {x: i for i, x in enumerate(tms)}
        sym_embedding, tm_tensors = [], []
        for tm in embeddings:
            tm_tensors += [tm_to_idx[tm] for _ in range(len(embeddings[tm]))]
            sym_embedding += embeddings[tm]

        footprints = torch.stack(sym_embedding, dim=0)
        tm_tensors = torch.tensor(tm_tensors)
        tsne = TSNE(2, verbose=1)
        tsne_proj = tsne.fit_transform(footprints)
        # Plot those points as a scatter plot and label them based on the pred labels

        tm_5 = self.plot_fig(embeddings, tm_tensors, tm_to_idx, tsne_proj, 5)
        tm_10 = self.plot_fig(embeddings, tm_tensors, tm_to_idx, tsne_proj, 10)
        tm_20 = self.plot_fig(embeddings, tm_tensors, tm_to_idx, tsne_proj, 20)
        tm_30 = self.plot_fig(embeddings, tm_tensors, tm_to_idx, tsne_proj, 30)
        tm_50 = self.plot_fig(embeddings, tm_tensors, tm_to_idx, tsne_proj, 50)
        columns = ['5TMs', '10TMs', '20TMs', '30TMs', '50TMs']
        wb_table = wandb.Table(columns=columns)
        wb_table.add_data(wandb.Image(tm_5), wandb.Image(tm_10), wandb.Image(tm_20), wandb.Image(tm_30),
                          wandb.Image(tm_50))
        wandb.log({f'{viz_name}': wb_table}, step=self._current_step)

    def _validate(self, i_epoch, val_loader, mode='val'):
        val_start_time = time.time()
        # set model to eval
        self._model.set_eval()
        val_errors = {}
        val_dict = {}
        all_accuracies = {}
        for i_train_batch, train_batch in enumerate(val_loader):
            final_loss, log_info, accuracies = self._compute_loss(train_batch, log_data=i_train_batch == 0)
            for key in accuracies:
                if key not in all_accuracies:
                    all_accuracies[key] = []
                all_accuracies[key] += accuracies[key]

            # store current batch errors
            for k, v in log_info.items():
                if k in val_errors:
                    val_errors[k].append(v)
                else:
                    val_errors[k] = [v]

        for k in val_errors.keys():
            val_errors[k] = sum(val_errors[k]) / len(val_errors[k])

        now_time = time.strftime("%H:%M", time.localtime(val_start_time))
        for k, v in val_errors.items():
            output = "{} {} {}: Epoch [{}] Step [{}] loss {:.4f}".format(
                k, mode, now_time, i_epoch, self._current_step, val_errors[k])
            print(output)
            val_dict[f'{mode}/{k}'] = val_errors[k]

        val_dict[f'{mode}/loss'] =\
            sum([self._model.normalize_lambda(t) * val_dict[f'{mode}/loss_{t}'] for t in args.tasks])

        for key in all_accuracies:
            val_dict[f'{mode}/acc/{key}'] = sum(all_accuracies[key]) / len(all_accuracies[key])

        # set model back to train
        self._model.set_train()
        wandb.log(val_dict, step=self._current_step)
        return val_dict

    def _display_terminal(self, iter_start_time, i_epoch, i_train_batch, num_batches, train_dict):
        errors = train_dict
        t = (time.time() - iter_start_time)
        start_time = time.strftime("%H:%M", time.localtime(iter_start_time))
        output = "Time {}\tBatch Time {:.2f}\t Epoch [{}]([{}/{}])\t loss {:.4f}\t".format(
            start_time, t,
            i_epoch, i_train_batch, num_batches,
            errors['loss'])
        for task in args.tasks:
            output += 'loss_{} {:.4f}\t'.format(task, errors['loss_{}'.format(task)])
        print(output)


if __name__ == "__main__":
    trainer = Trainer()
