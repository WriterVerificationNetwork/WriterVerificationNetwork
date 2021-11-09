import os

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from dataset.utils import MAX_WIDTH, MAX_HEIGHT, bincount_app, resize_image


class ImagePairFolder(Dataset):

    def __init__(self, original_folder_dir, ground_truth_folder_dir, data_transform, max_img_w, max_img_h):
        # Init folder dir
        self.original_folder_dir = original_folder_dir
        self.ground_truth_folder_dir = ground_truth_folder_dir
        self.data_transform = data_transform
        self.max_img_w = max_img_w
        self.max_img_h = max_img_h

        # Create image item
        self.image_list = []
        for root, dirs, files in os.walk(self.ground_truth_folder_dir):
            for file in files:
                if file != ".DS_Store" not in file:
                    try:
                        item = {
                            "file_name": file,
                            "letter": file[0],
                            "TM": file.split("_")[1]
                        }
                        self.image_list.append(item)
                    except Exception:
                        print(file)

        # Generate image pair
        self.image_pair_list = [(a, b) for idx, a in enumerate(self.image_list) for b in self.image_list[idx + 1:]
                                if a["TM"] == b["TM"] and a["letter"] == b["letter"]]

    def __getitem__(self, idx):
        # Get first image and its ground truth
        first_img_tensor = get_image(os.path.join(self.original_folder_dir,
                                                  self.image_pair_list[idx][0]["letter"],
                                                  self.image_pair_list[idx][0]["file_name"]),
                                     self.data_transform, "1")
        first_img_gt_tensor = get_image(os.path.join(self.ground_truth_folder_dir,
                                                     self.image_pair_list[idx][0]["file_name"]),
                                        self.data_transform, "2")

        # Get second image and its ground truth
        second_img_tensor = get_image(os.path.join(self.original_folder_dir,
                                                   self.image_pair_list[idx][1]["letter"],
                                                   self.image_pair_list[idx][1]["file_name"]),
                                      self.data_transform, "3")
        second_img_gt_tensor = get_image(os.path.join(self.ground_truth_folder_dir,
                                                      self.image_pair_list[idx][1]["file_name"]),
                                         self.data_transform, "4")

        return first_img_tensor, first_img_gt_tensor, second_img_tensor, second_img_gt_tensor

    def __len__(self) -> int:
        return len(self.image_list)


def get_image(image_path, data_transform, name):
    with Image.open(image_path) as img:
        # Resize image to make sure image size is smaller than page size
        width, height = img.size
        ratio_w = MAX_WIDTH / width
        ratio_h = MAX_HEIGHT / height
        scale = min(ratio_h, ratio_w)
        image = resize_image(img, scale)
        # Find the dominant color
        dominant_color = bincount_app(np.asarray(img.convert("RGB")))
        # Add image to the background
        padding_image = Image.new(mode="RGB", size=(MAX_WIDTH, MAX_HEIGHT), color=dominant_color)
        padding_image.paste(image, box=(0, 0))
        padding_image.save(name + ".png")

    # Transform image
    img_tensor = data_transform(padding_image)

    return img_tensor
