import os

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.filters import threshold_otsu

from dataset.utils import ORIGINAL_FOLDER_DIR, GT_DENOISING_DATA_DIRNAME, GOOD_GT_DATA_DIRNAME, AVERAGE_GT_DATA_DIRNAME, \
    BAD_GT_DATA_DIRNAME


def create_ground_truth():
    index = 0
    for root, dirs, files in os.walk(ORIGINAL_FOLDER_DIR):
        for file in files:
            if file != ".DS_Store" not in file:
                try:
                    thresh = binarize_image_Otsu_denoise(os.path.join(root, file))
                    if not os.path.exists(os.path.join(GT_DENOISING_DATA_DIRNAME, file[0])):
                        os.makedirs(os.path.join(GT_DENOISING_DATA_DIRNAME, file[0]))
                    cv2.imwrite(os.path.join(GT_DENOISING_DATA_DIRNAME, file[0], file), thresh)
                    index += 1
                    print(index)
                    origin = cv2.imread(os.path.join(root, file))
                    binarization = cv2.imread(os.path.join(GT_DENOISING_DATA_DIRNAME, file[0], file))
                    # concatanate image Horizontally
                    hori = np.concatenate((origin, binarization), axis=1)
                    cv2.imshow('result', hori)
                    key = cv2.waitKey(0)
                    #  0
                    if key == 48:
                        print("good")
                        if not os.path.exists(GOOD_GT_DATA_DIRNAME):
                            os.makedirs(GOOD_GT_DATA_DIRNAME)
                        save_image(GOOD_GT_DATA_DIRNAME, os.path.join(GT_DENOISING_DATA_DIRNAME, file[0], file))
                    # 1
                    elif key == 49:
                        print("average")
                        if not os.path.exists(AVERAGE_GT_DATA_DIRNAME):
                            os.makedirs(AVERAGE_GT_DATA_DIRNAME)
                        save_image(AVERAGE_GT_DATA_DIRNAME, os.path.join(GT_DENOISING_DATA_DIRNAME, file[0], file))
                    # 2
                    elif key == 50:
                        print("bad")
                        if not os.path.exists(BAD_GT_DATA_DIRNAME):
                            os.makedirs(BAD_GT_DATA_DIRNAME)
                        save_image(BAD_GT_DATA_DIRNAME, os.path.join(GT_DENOISING_DATA_DIRNAME, file[0], file))
                    cv2.destroyAllWindows()
                except Exception as e:
                    print(e)


def save_image(folder, binarized_image_path):
    temp = Image.open(binarized_image_path)
    temp.save(os.path.join(folder, os.path.basename(binarized_image_path)))


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
    masked_image = Image.fromarray(masked_image)
    masked_image.save("convert.png")
    # Binarize image
    image = np.asarray(Image.open("convert.png").convert('L'))
    t_otsu = threshold_otsu(image)
    binary_image = image > t_otsu
    indices = binary_image.astype(np.uint8)  # convert to an unsigned byte
    indices *= 255
    img = Image.fromarray(indices, mode='L')
    img = ImageOps.invert(img)
    img.save("binarize.png")
    # Denoise image
    img = cv2.imread("binarize.png", 0)
    blur = cv2.GaussianBlur(img, (13, 13), 0)
    thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
    return thresh


if __name__ == "__main__":
    create_ground_truth()
