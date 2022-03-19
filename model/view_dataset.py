import numpy as np
import torchvision
from matplotlib import pyplot as plt

from dataset.tm_dataset import TMDataset
from options.train_options import TrainOptions
from utils.transform import get_transforms

args = TrainOptions().parse()

transforms = get_transforms(args)
untensor = torchvision.transforms.ToPILImage()

dataset_train = TMDataset(args.gt_dir, args.gt_binarized_dir, args.filter_file, transforms, split_from=0, split_to=1,
                          min_n_sample_per_letter=args.min_n_sample_per_letter,
                          min_n_sample_per_class=args.min_n_sample_per_class)

fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion() # Turns interactive mode on (probably unnecessary)
fig.show() # Initially shows the figure

for idx, item in enumerate(dataset_train):
    # a = 1
    viewer.clear()
    img = np.asarray(untensor(item['img_anchor']))
    viewer.imshow(img)

    plt.pause(0.05)
    fig.canvas.draw()