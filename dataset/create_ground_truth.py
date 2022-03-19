import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.filters import threshold_otsu

from options.train_options import TrainOptions


def create_ground_truth(image_dir, bin_dir):
    os.makedirs(bin_dir, exist_ok=True)
    images = glob.glob(os.path.join(image_dir, f'*.png'))
    for file in images:
        final_img = binarize_image_Otsu_denoise(file)
        file_name = os.path.basename(file)
        cv2.imwrite(os.path.join(bin_dir, file_name), final_img)


def binarize_image_Otsu_denoise(image_path):
    image_array = np.asarray(Image.open(image_path).convert('RGB'))
    # Get the dominant color
    # dominant_color = bincount_app(image_array)
    dominant_color = (165, 134, 105)
    # Convert white pixel to dominant color pixel
    white_color = (190, 190, 190)
    temp = 65
    lower = np.array([white_color[0] - temp, white_color[1] - temp, white_color[2] - temp])
    upper = np.array([white_color[0] + temp, white_color[1] + temp, white_color[2] + temp])
    mask = cv2.inRange(image_array, lower, upper)
    masked_image = np.copy(image_array)
    masked_image[mask != 0] = dominant_color
    # masked_image = Image.fromarray(masked_image)
    # masked_image.save("convert.png")
    # Binarize image
    image = np.asarray(Image.open("convert.png").convert('L'))
    t_otsu = threshold_otsu(image)
    binary_image = image > t_otsu
    indices = binary_image.astype(np.uint8)  # convert to an unsigned byte
    indices *= 255
    img = Image.fromarray(indices, mode='L')
    img = ImageOps.invert(img)
    # Denoise image
    img = np.array(img)
    blur = cv2.GaussianBlur(img, (13, 13), 0)
    final_img = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
    return final_img


if __name__ == "__main__":
    args = TrainOptions().parse()
    create_ground_truth(args.gt_dir, args.gt_binarized_dir)
