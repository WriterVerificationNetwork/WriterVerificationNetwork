import numpy as np
from PIL import Image



ORIGINAL_FOLDER_DIR = "/Users/hongtruong/Documents/Collaboration_Basel_Bordeaux/bt1_by_letters_20210824"
GT_DENOISING_DATA_DIRNAME = "/Users/hongtruong/Documents/Collaboration_Basel_Bordeaux/bt1_by_letters_20210824_ground_truth_denoising"
GROUND_TRUTH_FOLDER_DIR = "/Users/hongtruong/Documents/Collaboration_Basel_Bordeaux/bt1_by_letters_binarization"

GOOD_GT_DATA_DIRNAME = GROUND_TRUTH_FOLDER_DIR + "/good"
AVERAGE_GT_DATA_DIRNAME = GROUND_TRUTH_FOLDER_DIR + "/average"
BAD_GT_DATA_DIRNAME = GROUND_TRUTH_FOLDER_DIR + "/bad"

letters = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'ϲ', 'τ', 'υ', 'φ',
           'χ', 'ψ', 'ω']

letter_to_idx = {x: i for i, x in enumerate(letters)}
idx_to_letter = {i: x for i, x in enumerate(letters)}


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator


# function get dominant color
def bincount_app(image_array):
    a2D = image_array.reshape(-1, image_array.shape[-1])
    col_range = (256, 256, 256)  # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    return image.resize((round(image.width * scale_factor), round(image.height * scale_factor)),
                        resample=Image.BILINEAR)
