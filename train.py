import numpy as np
import torch
from torchvision import transforms

from dataset.image_pair_folder import ImagePairFolder
from dataset.utils import GROUND_TRUTH_FOLDER_DIR, ORIGINAL_FOLDER_DIR, MAX_WIDTH, MAX_HEIGHT

if __name__ == "__main__":
    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed)

    #### Preparing dataset
    data_transform = transforms.Compose([transforms.ToTensor()])
    # Creating pair image from original dataset
    image_dataset = ImagePairFolder(ORIGINAL_FOLDER_DIR, GROUND_TRUTH_FOLDER_DIR, data_transform, MAX_WIDTH, MAX_HEIGHT)
    image_dataset[5]
