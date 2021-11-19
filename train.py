import gc
import statistics
import time

import torch
import wandb

from dataset.data_loader import WriterDataLoader
from dataset.image_dataset import ImageDataset
from model.model_factory import ModelsFactory
from options.train_options import TrainOptions
from utils.misc import EarlyStop
from utils.transform import get_transforms
from utils.wb_utils import log_prediction

args = TrainOptions().parse()


class Trainer:
    def __init__(self):
        wandb.init(group=args.group,
                   project=args.wb_project,
                   entity=args.wb_entity)
        wandb.run.name = args.name
        wandb.run.save()
        wandb.config.update(args)
        device = torch.device('cuda' if args.cuda else 'cpu')
        self._model = ModelsFactory.get_model(args, is_train=True, device=device, dropout=0.4)
        transforms = get_transforms(args)
        dataset_train = ImageDataset(args.gt_dir, args.gt_binarized_dir, args.filter_file, transforms,
                                     split_from=0, split_to=0.8)
        self._model.init_losses('Train', args.use_weighted_loss, dataset_train)
        self.data_loader_train = WriterDataLoader(dataset_train, is_train=True, numb_threads=args.n_threads_train,
                                                  batch_size=args.batch_size)

        dataset_val = ImageDataset(args.gt_dir, args.gt_binarized_dir, args.filter_file, transforms,
                                   split_from=0.8, split_to=1)
        self._model.init_losses('Val', use_weighted_loss=False, dataset=dataset_val)
        self.data_loader_val = WriterDataLoader(dataset_val, is_train=False, numb_threads=args.n_threads_train,
                                                batch_size=args.batch_size)

        self.early_stop = EarlyStop(args.early_stop)
        print("Training tasks {}".format(args.tasks))
        print("Training sets: {} images".format(len(dataset_train)))
        print("Validating sets: {} images".format(len(dataset_val)))

        self._train()

    def _train(self):
        self._current_step = 0
        self._last_save_time = time.time()
        best_val_acc = 0.
        for i_epoch in range(1, args.nepochs + 1):
            epoch_start_time = time.time()
            self._model.get_current_lr()
            # train epoch
            self._train_epoch(i_epoch)
            if args.lr_policy == 'step':
                self._model.lr_scheduler.step()

            if not i_epoch % args.n_epochs_per_eval == 0:
                continue

            val_dict = self._validate(i_epoch)
            gc.collect()

            current_acc = val_dict['val/acc/footprint']
            if current_acc > best_val_acc:
                print("Footprint val acc improved, from {:.4f} to {:.4f}".format(best_val_acc, current_acc))
                best_val_acc = current_acc
                for key in val_dict:
                    wandb.run.summary[f'best_model/{key}'] = val_dict[key]
                self._model.save()  # save best model

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, args.nepochs, time_epoch, time_epoch / 60, time_epoch / 3600))

            if self.early_stop.should_stop(1 - current_acc):
                print(f'Early stop at epoch {i_epoch}')
                break

    def __get_data(self, batch_data, image_key, reconstruct_key, symbol_key):
        res = {'image': batch_data[image_key]}
        if 'reconstruct' in args.tasks:
            res['reconstruct'] = batch_data[reconstruct_key]
        if 'symbol' in args.tasks:
            res['symbol'] = batch_data[symbol_key]
        return res

    def _compute_loss(self, batch_data, log_data=False, n_log_items=10):
        input_data = self.__get_data(batch_data, 'img_anchor', 'bin_anchor', 'symbol')
        anchor_out, anchor_loss, anchor_log = self._model.compute_loss(input_data)
        input_data = self.__get_data(batch_data, 'img_positive', 'bin_positive', 'symbol')
        pos_out, pos_loss, pos_log = self._model.compute_loss(input_data)
        input_data = self.__get_data(batch_data, 'img_negative', 'bin_negative', 'symbol')
        neg_out, neg_loss, neg_log = self._model.compute_loss(input_data)
        log_info = {k: statistics.mean([anchor_log[k], pos_log[k], neg_log[k]]) for k in anchor_log}

        final_losses = [anchor_loss, pos_loss, neg_loss]
        footprint_loss = self._model.compute_footprint(anchor_out['footprint'], pos_out['footprint'],
                                                       neg_out['footprint'])
        final_losses += [footprint_loss * 4.]
        log_info['loss_footprint'] = footprint_loss.item()

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
            wandb.log({'val_prediction': wb_table})

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

        final_losses = sum(final_losses) / len(final_losses)
        log_info['loss'] = final_losses.item()
        return final_losses, log_info, accuracies

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

    def _validate(self, i_epoch):
        val_start_time = time.time()
        # set model to eval
        self._model.set_eval()
        val_errors = {}
        val_dict = {}
        data_loader = self.data_loader_val.get_dataloader()
        all_accuracies = {}
        for i_train_batch, train_batch in enumerate(data_loader):
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
            output = "{} Validation {}: Epoch [{}] Step [{}] loss {:.4f}".format(
                k, now_time, i_epoch, self._current_step, val_errors[k])
            print(output)
            val_dict[f'val/{k}'] = val_errors[k]

        val_dict[f'val/loss'] =\
            sum([self._model.normalize_lambda(t) * val_dict[f'val/loss_{t}'] for t in args.tasks])

        for key in all_accuracies:
            val_dict[f'val/acc/{key}'] = sum(all_accuracies[key]) / len(all_accuracies[key])

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
