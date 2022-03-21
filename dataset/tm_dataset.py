import glob
import json
import os
import random

import imagesize
import numpy as np
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.utils import resize_image, letters, MAX_WIDTH, MAX_HEIGHT, letter_to_idx


class TMDataset(Dataset):

    def __init__(self, gt_dir, gt_binarized_dir, filter_file, transforms, split_from, split_to,
                 unfold=False, min_n_sample_per_letter=0, min_n_sample_per_class=0,
                 without_imgs=False, training_mode=False):
        # Init folder dir
        self.gt_dir = gt_dir
        self.gt_binarized_dir = gt_binarized_dir
        self.transforms = transforms
        self.without_imgs = without_imgs
        self.anchor_tms = []
        self.training_mode = training_mode

        # Create image item
        temp_image_list = []
        tm_map = {}
        letter_tm_map = {}
        total_imgs_removing_by_letter = 0
        for letter in letters:
            image_by_letter = glob.glob(os.path.join(self.gt_dir, f'{letter}_*.png'))
            if len(image_by_letter) < min_n_sample_per_letter:
                # Ignore if number of samples per letter is less than min_n_sample_per_letter
                total_imgs_removing_by_letter += len(image_by_letter)
                continue
            image_by_letter = sorted(image_by_letter)
            letter_tm_map[letter] = {}
            for img in image_by_letter:
                width, height = imagesize.get(img)
                if width < 32 and height < 32:
                    continue
                tm = os.path.basename(img).split("_")[1]
                if tm not in tm_map:
                    tm_map[tm] = []
                if tm not in letter_tm_map[letter]:
                    letter_tm_map[letter][tm] = []
                tm_map[tm].append(img)
                letter_tm_map[letter][tm].append(img)
            sp_from, sp_to = int(len(image_by_letter) * split_from), int(len(image_by_letter) * split_to)
            temp_image_list.append(image_by_letter[sp_from: sp_to])
        tm_filter = set([k for k, v in tm_map.items() if len(v) < min_n_sample_per_class])
        total_imgs_removing_by_class = sum([len(v) for k, v in tm_map.items() if len(v) < min_n_sample_per_class])
        print(f'Total number of images removed by letter lever filter: {total_imgs_removing_by_letter}')
        print(f'Total number of images removed by class lever filter: {total_imgs_removing_by_class}')

        positive_tms = {x: [x] for x in tm_map.keys()}
        negative_tms = {x: [] for x in tm_map.keys()}
        with open(filter_file) as f:
            triplet_filter = json.load(f)

        for item in triplet_filter['relations']:
            current_tm = item['category']
            if current_tm not in tm_map:
                print(f'TM {current_tm} is not available on the training dataset')
                continue
            for second_item in item['relations']:
                second_tm = second_item['category']
                if second_tm not in tm_map:
                    print(f'TM {second_tm} is not available on the training dataset')
                    continue
                relationship = second_item['relationship']
                if current_tm == '' or second_tm == '':
                    continue
                if relationship == 4:
                    negative_tms[current_tm] += [second_tm]
                    negative_tms[second_tm] += [current_tm]
                if relationship == 1:
                    positive_tms[current_tm] += [second_tm]
                    positive_tms[second_tm] += [current_tm]

        # same_categories = ['60764', '60891', '60842', '60934']
        # for tm in same_categories:
        #     for tm2 in same_categories:
        #         positive_tms[tm].add(tm2)

        self.image_list = []
        for image_by_letter in tqdm(temp_image_list):
            for anchor in image_by_letter:
                anchor_tm = os.path.basename(anchor).split("_")[1]
                if anchor_tm in tm_filter:
                    continue
                if anchor_tm not in negative_tms:
                    continue
                anchor_letter = os.path.basename(anchor).split("_")[0]
                img_negative_tms = set(letter_tm_map[anchor_letter].keys()).intersection(negative_tms[anchor_tm])
                if len(img_negative_tms) > 0:
                    if not unfold:
                        self.image_list.append((positive_tms[anchor_tm], anchor, img_negative_tms))
                        self.anchor_tms.append(anchor_tm)
                    else:
                        for neg_tm in img_negative_tms:
                            for pos_tm in set(positive_tms[anchor_tm]):
                                self.image_list.append(([pos_tm], anchor, [neg_tm]))
                                self.anchor_tms.append(anchor_tm)
        self.letter_tm_map = letter_tm_map

    def __getitem__(self, idx):
        # anchor
        positive_tms, anchor_img, negative_tms = self.image_list[idx]
        anchor = os.path.basename(anchor_img)
        anchor_letter = anchor.split("_")[0]

        positive_tm = random.choice(list(positive_tms))
        positive_img = random.choice(self.letter_tm_map[anchor_letter][positive_tm])
        negative_tm = random.choice(list(negative_tms))
        negative_img = random.choice(self.letter_tm_map[anchor_letter][negative_tm])

        # anchor image
        moving_percent = random.randint(0, 10) / 10.
        anchor = os.path.basename(anchor_img)
        anchor_tm = anchor.split("_")[1]
        if self.without_imgs:
            return {
                'symbol': letter_to_idx[anchor.split("_")[0]],
                'anchor_path': anchor,
                'tm_anchor': anchor_tm
            }

        should_flip = np.random.choice(np.arange(0, 2), p=[1 - 0.5, 0.5])
        should_mirror = np.random.choice(np.arange(0, 2), p=[1 - 0.5, 0.5])
        img_anchor, origin_anc = get_image(os.path.join(self.gt_dir, anchor), self.transforms, is_bin_img=False,
                               mov=moving_percent, flip=should_flip, mirror=should_mirror)
        bin_anchor, _ = get_image(os.path.join(self.gt_binarized_dir, anchor), self.transforms, is_bin_img=True,
                                  mov=moving_percent, flip=should_flip, mirror=should_mirror, org_img=origin_anc)

        # positive image
        moving_percent = random.randint(0, 10) / 10.
        img = os.path.basename(positive_img)
        img_positive, origin_pos = get_image(os.path.join(self.gt_dir, img), self.transforms, is_bin_img=False,
                                 mov=moving_percent, flip=should_flip, mirror=should_mirror)
        bin_positive, _ = get_image(os.path.join(self.gt_binarized_dir, img), self.transforms, is_bin_img=True,
                                    mov=moving_percent, flip=should_flip, mirror=should_mirror, org_img=origin_pos)

        should_replace = np.random.choice(np.arange(0, 2), p=[1 - 0.5, 0.5])
        if self.training_mode and should_replace:
            img_positive = bin_positive

        # negative image
        moving_percent = random.randint(0, 10) / 10.
        img = os.path.basename(negative_img)
        img_negative, origin_neg = get_image(os.path.join(self.gt_dir, img), self.transforms, is_bin_img=False,
                                 mov=moving_percent, flip=should_flip, mirror=should_mirror)
        bin_negative, _ = get_image(os.path.join(self.gt_binarized_dir, img), self.transforms, is_bin_img=True,
                                    mov=moving_percent, flip=should_flip, mirror=should_mirror, org_img=origin_neg)

        should_replace = np.random.choice(np.arange(0, 2), p=[1 - 0.5, 0.5])
        if self.training_mode and should_replace:
            img_negative = bin_negative

        return {
            'symbol': letter_to_idx[anchor.split("_")[0]],
            'img_anchor': img_anchor,
            'anchor_path': anchor,
            'tm_anchor': anchor_tm,
            'bin_anchor': bin_anchor,
            'img_positive': img_positive,
            'bin_positive': bin_positive,
            'img_negative': img_negative,
            'bin_negative': bin_negative
        }

    def __len__(self) -> int:
        return len(self.image_list)


def get_image(image_path, data_transform, is_bin_img=False, mov=0., flip=False, mirror=False, org_img=None):
    with Image.open(image_path) as img:
        # Resize image to make sure image size is smaller than page size
        width, height = img.size
        ratio_w = MAX_WIDTH / width
        ratio_h = MAX_HEIGHT / height
        scale = min(ratio_h, ratio_w)
        image = resize_image(img, scale).convert('RGB')
        width, height = image.size
        if not is_bin_img:
            image = ImageOps.invert(image)
        # Find the dominant color
        # dominant_color = bincount_app(np.asarray(img.convert("RGB")))
        # Add image to the background
        padding_image = Image.new(mode="RGB", size=(MAX_WIDTH, MAX_HEIGHT), color=(0, 0, 0))
        padding_image.paste(image, box=(int(mov * (MAX_WIDTH - width)), int(mov * (MAX_HEIGHT - height))))
        if flip:
            padding_image = ImageOps.flip(padding_image)
        if mirror:
            padding_image = ImageOps.mirror(padding_image)
        # if is_bin_img:
        #     padding_image = padding_image.convert("L")
        if org_img is not None and is_bin_img:
            bin_img = np.asarray(padding_image) / 255.
            padding_image = bin_img * np.asarray(org_img)
            padding_image = Image.fromarray(padding_image.astype(np.uint8))

    # Transform image
    # if not is_bin_img:
    #     img_tensor = data_transform(padding_image)
    # else:
    #     img_tensor = torchvision.transforms.ToTensor()(padding_image)
    img_tensor = data_transform(padding_image)

    return img_tensor, padding_image
