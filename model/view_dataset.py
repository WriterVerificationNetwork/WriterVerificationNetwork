import matplotlib
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
# matplotlib.use('MACOSX')
matplotlib.use('TkAgg')

from dataset.tm_dataset import TMDataset
from options.train_options import TrainOptions
from utils.transform import get_transforms, reverse_transform

args = TrainOptions().parse()

transforms = get_transforms(args)
untensor = reverse_transform()

dataset_train = TMDataset(args.gt_dir, args.gt_binarized_dir, args.filter_file, transforms, split_from=0, split_to=1,
                          min_n_sample_per_letter=args.min_n_sample_per_letter,
                          min_n_sample_per_class=args.min_n_sample_per_class, training_mode=True)

fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion() # Turns interactive mode on (probably unnecessary)
fig.show() # Initially shows the figure

for idx, item in enumerate(dataset_train):
    # a = 1
    viewer.clear()
    img = item['img_anchor']
    img_bin = item['bin_anchor']
    img_pos = item['img_positive']
    img_pos_bin = item['bin_positive']
    img_neg = item['img_negative']
    img_neg_bin = item['bin_negative']

    img = torch.cat([img, img_bin], dim=1)
    img_neg = torch.cat([img_neg, img_neg_bin], dim=1)
    img_pos = torch.cat([img_pos, img_pos_bin], dim=1)

    out_img = torch.cat([img, img_pos, img_neg], dim=2)
    viewer.imshow(np.asarray(untensor(out_img)))

    plt.pause(2)
    fig.canvas.draw()