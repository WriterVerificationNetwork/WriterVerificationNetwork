import os

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.filters import threshold_otsu


from dataset.utils import INPUT_DATA_DIRNAME, GT_DENOISING_DATA_DIRNAME

test_file_02 = "/Users/hongtruong/Documents/Collaboration_Basel_Bordeaux/bt1_by_letters_20210824/α/α_60215_bt1_Iliad.9.186.15.png"
test_file_01 = "/Users/hongtruong/Documents/Collaboration_Basel_Bordeaux/bt1_by_letters_20210824/α/α_60215_bt1_Iliad.9.181.5.png"
test_file_03 = "/Users/hongtruong/Documents/Collaboration_Basel_Bordeaux/bt1_by_letters_20210824/α/α_60215_bt1_Iliad.9.184.19.png"
test_file_04 = "/Users/hongtruong/Documents/Collaboration_Basel_Bordeaux/bt1_by_letters_20210824/α/α_60220_bt1_Iliad.5.74.1.png"
test_file_05 = "/Users/hongtruong/Documents/Collaboration_Basel_Bordeaux/bt1_by_letters_20210824/α/α_60220_bt1_Iliad.5.81.5.png"


def create_ground_truth():
    for root, dirs, files in os.walk(INPUT_DATA_DIRNAME):
        for file in files:
            if file != ".DS_Store" and "602014" not in file:
                try:
                    thresh = binarize_image_Otsu_denoise(os.path.join(root, file))
                    if not os.path.exists(os.path.join(GT_DENOISING_DATA_DIRNAME, file[0])):
                        os.makedirs(os.path.join(GT_DENOISING_DATA_DIRNAME, file[0]))
                    cv2.imwrite(os.path.join(GT_DENOISING_DATA_DIRNAME, file[0], file), thresh)
                except Exception:
                    print(file)


# function get dominant color
def bincount_app(image_array):
    a2D = image_array.reshape(-1, image_array.shape[-1])
    col_range = (256, 256, 256)  # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)



def binarize_image_Otsu_denoise(image_path):
    image_array = np.asarray(Image.open(image_path).convert('RGB'))
    # Get the dominant color
    dominant_color = bincount_app(image_array)
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
    # create_ground_truth()

    thresh = binarize_image_Otsu_denoise(test_file_03)
    cv2.imwrite("test_denoise_03_0.png", thresh)
