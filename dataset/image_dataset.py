import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import openpyxl

from dataset.utils import bincount_app, resize_image, letters, MAX_WIDTH, MAX_HEIGHT, \
    ORIGINAL_FOLDER_DIR, GOOD_GT_DATA_DIRNAME, FILTER_TM_IMAGE_FILE_PATH, problematic_tm, MAX_BIN_WIDTH, MAX_BIN_HEIGHT


class ImageDataset(Dataset):

    def __init__(self, gt_dir, gt_binarized_dir, transforms):
        # Init folder dir
        self.gt_dir = gt_dir
        self.gt_binarized_dir = gt_binarized_dir
        self.transforms = transforms

        # Create image item
        temp_image_list = []
        for letter in letters:
            image_by_letter = []
            for root, dirs, files in os.walk(self.gt_binarized_dir):
                for file in files:
                    if file.endswith(".png") and file.startswith(letter):
                        image_by_letter.append(file)
            temp_image_list.append(image_by_letter)

        negative_tm_list = create_negative_tm_pair()
        self.image_list = []
        for image_by_letter in temp_image_list:
            positive_image_list = []
            negative_image_list = []
            for anchor in image_by_letter:
                for img in image_by_letter:
                    if anchor.split("_")[1] not in problematic_tm and img.split("_")[1] not in problematic_tm:
                        if anchor != img and anchor.split("_")[1] == img.split("_")[1]:
                            positive_image_list.append([anchor, img])
                        else:
                            if [anchor.split("_")[1], img.split("_")[1]] in negative_tm_list:
                                negative_image_list.append([anchor, img])
            for pos_image in positive_image_list:
                for neg_image in negative_image_list:
                    if pos_image[1] == neg_image[0]:
                        self.image_list.append([pos_image[0], pos_image[1], neg_image[1]])

    def __getitem__(self, idx):
        # anchor
        anchor = self.image_list[idx][0]
        img_anchor = get_image(os.path.join(self.gt_dir, anchor.split("_")[0], anchor),
                               self.transforms,
                               MAX_WIDTH,
                               MAX_HEIGHT,
                               False,
                               "img_anchor")
        bin_anchor = get_image(os.path.join(self.gt_binarized_dir, anchor),
                               self.transforms,
                               MAX_BIN_WIDTH,
                               MAX_BIN_HEIGHT,
                               True,
                               "bin_anchor")

        # positive image
        img = self.image_list[idx][1]
        img_positive = get_image(os.path.join(self.gt_dir, img.split("_")[0], img),
                                 self.transforms,
                                 MAX_WIDTH,
                                 MAX_HEIGHT,
                                 False,
                                 "img_positive")
        bin_positive = get_image(os.path.join(self.gt_binarized_dir, img),
                                 self.transforms,
                                 MAX_BIN_WIDTH,
                                 MAX_BIN_HEIGHT,
                                 True,
                                 "bin_positive")

        # negative image
        img = self.image_list[idx][2]
        img_negative = get_image(os.path.join(self.gt_dir, img.split("_")[0], img),
                                 self.transforms,
                                 MAX_WIDTH,
                                 MAX_HEIGHT,
                                 False,
                                 "img_negative")
        bin_negative = get_image(os.path.join(self.gt_binarized_dir, img),
                                 self.transforms,
                                 MAX_BIN_WIDTH,
                                 MAX_BIN_HEIGHT,
                                 True,
                                 "bin_negative")

        return {
            'symbol': anchor.split("_")[0],
            'img_anchor': img_anchor,
            'bin_anchor': bin_anchor,
            'img_positive': img_positive,
            'bin_positive': bin_positive,
            'img_negative': img_negative,
            'bin_negative': bin_negative
        }

    def __len__(self) -> int:
        return len(self.image_list)


def get_image(image_path, data_transform, max_w, max_h, is_bin_img=False, name=""):
    with Image.open(image_path) as img:
        # Resize image to make sure image size is smaller than page size
        width, height = img.size
        ratio_w = max_w / width
        ratio_h = max_h / height
        scale = min(ratio_h, ratio_w)
        image = resize_image(img, scale)
        # Find the dominant color
        dominant_color = bincount_app(np.asarray(img.convert("RGB")))
        # Add image to the background
        padding_image = Image.new(mode="RGB", size=(max_w, max_h), color=dominant_color)
        padding_image.paste(image, box=(0, 0))
        # padding_image.save(name + ".png")

        if is_bin_img:
            padding_image = padding_image.convert("L")

    # Transform image
    img_tensor = data_transform(padding_image)

    return img_tensor


def create_negative_tm_pair():
    filter_tm_file = Path(FILTER_TM_IMAGE_FILE_PATH)
    filter_tm_object = openpyxl.load_workbook(filter_tm_file)
    sheet = filter_tm_object.active
    negative_tm_list = []
    for i, row in enumerate(sheet.iter_rows(min_row=2, max_row=82, min_col=2, max_col=82, values_only=True)):
        for j, cell in enumerate(row):
            if cell == "X" and (str(2 + i) not in problematic_tm) and (str(2 + j) not in problematic_tm):
                negative_tm_list.append([str(sheet.cell(row=2 + i, column=1).value),
                                         str(sheet.cell(row=1, column=2 + j).value)])
                negative_tm_list.append([str(sheet.cell(row=1, column=2 + j).value),
                                         str(sheet.cell(row=2 + i, column=1).value)])
    return negative_tm_list


if __name__ == "__main__":
    start_time = datetime.now()
    dataset = ImageDataset(ORIGINAL_FOLDER_DIR,
                           GOOD_GT_DATA_DIRNAME,
                           torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    print(len(dataset.image_list))
    end_time = datetime.now()
    print(end_time-start_time)
    print(dataset[0])
