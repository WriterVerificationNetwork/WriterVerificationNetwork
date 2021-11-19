import torch
import wandb
from torch import nn

from dataset.utils import idx_to_letter
from utils.transform import reverse_transform


def wb_img(image):
    return wandb.Image(reverse_transform()(image))


def log_prediction(wb_table, log_counter, anchor, bin_anchor, positive, negative, symbol, anchor_out, pos_out,
                   neg_out, n_items, bin_weight):
    anchor = anchor[:n_items].cpu()
    anchor_bin_pred = anchor_out['reconstruct'][:n_items].cpu() / bin_weight
    anchor_bin = bin_anchor[:n_items].cpu()
    positive = positive[:n_items].cpu()
    negative = negative[:n_items].cpu()
    symbol = symbol[:n_items].cpu().numpy()
    symbol_pred = torch.max(anchor_out['symbol'][:n_items], dim=1).indices.cpu().numpy()
    distance_func = nn.MSELoss(reduction='none')
    pos_distance = distance_func(anchor_out['footprint'][:n_items], pos_out['footprint'][:n_items]).mean(dim=1)
    neg_distance = distance_func(anchor_out['footprint'][:n_items], neg_out['footprint'][:n_items]).mean(dim=1)

    _id = 0
    for a, ab, abp, p, n, s, sp, pos, neg in zip(anchor, anchor_bin, anchor_bin_pred, positive, negative, symbol,
                                                 symbol_pred, pos_distance, neg_distance):
        img_id = str(_id) + "_" + str(log_counter)
        s = idx_to_letter[s]
        sp = idx_to_letter[sp]
        wb_table.add_data(img_id, wb_img(a), wb_img(ab), wb_img(abp), wb_img(p), wb_img(n), s, sp, pos, neg)
        _id += 1
