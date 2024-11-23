import random
import os

import numpy as np
from PIL import Image
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

GRID_DIMS = (18, 11)
batch_size = 64
# load the mean and std images
imageDirName = "../DATA/image"
# compute the train image statistics
mean_image = np.mean([np.load(os.path.join(imageDirName, "mean_train_img_2017.npy"))], axis=0)
std_image = np.mean([np.load(os.path.join(imageDirName, "std_train_img_2017.npy"))], axis=0)
# compute the pixels statistics
MEAN_PIX = np.mean(np.reshape(mean_image, (-1,3)), axis=0)         # (255, 255, 255) convention
STD_PIX = np.mean(np.reshape(std_image, (-1,3)), axis=0)           # (255, 255, 255) convention

# TRAIN image transformation pipeline
train_transformer = transforms.Compose([
    transforms.Resize(256),                                         # resize to (393, 256)
    transforms.Lambda(lambda img: img.crop(box=(0, 0, 256, 384))),  # crop to (384, 256)
    transforms.RandomHorizontalFlip(),                              # randomly flip image horizontally
    transforms.ToTensor(),                                          # transform it into a torch tensor and put in (1, 1, 1) convention
    transforms.Normalize(MEAN_PIX/255, (1,1,1))])                   # normalize the image

# EVAL image transformation pipeline (no horizontal flip)
eval_transformer = transforms.Compose([
    transforms.Resize(256),                                         # resize to (393, 256)
    transforms.Lambda(lambda img: img.crop(box=(0, 0, 256, 384))),  # crop to (384, 256)
    transforms.ToTensor(),                                          # transform it into a torch tensor and put in (1, 1, 1) convention
    transforms.Normalize(MEAN_PIX/255, (1,1,1))])                   # normalize the image

# EVAL image transformation pipeline for visualization (no normalization)
visual_transformer = transforms.Compose([
    transforms.Resize(256),                                         # resize to (393, 256)
    transforms.Lambda(lambda img: img.crop(box=(0, 0, 256, 384))),  # crop to (384, 256)
    transforms.ToTensor()])

class ClimbBinaryDataset(Dataset):
    def __init__(self, data_dir, split, n_dev=3*64):
        self.X = np.load(os.path.join(data_dir, "X_{}.npy".format(split)))
        self.y = np.load(os.path.join(data_dir, "y_{}.npy".format(split)))
        self.X = np.reshape(self.X, (-1, *GRID_DIMS))
        print(self.X.shape, self.y.shape, self.y.size)

    def __len__(self):
        return self.y.size

    def __getitem__(self, idx):
        x = from_numpy(self.X[idx]).long()
        return x, self.y[idx]
    
def fetch_dataloader(splits, params=0):
    dataloaders = {}
    n_examples = {}

    data_path = "../DATA/binary"

    for split in ['train', 'val', 'test']:
        if split == 'train':
            shuffle = True
        else:
            shuffle = False

        dataset = ClimbBinaryDataset(data_path, split, n_dev=batch_size)

        n_examples[split] = len(dataset)
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return dataloaders, n_examples