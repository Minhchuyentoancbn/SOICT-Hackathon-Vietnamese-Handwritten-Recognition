import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as F

import numpy as np
import math
from .config import LABEL_FILE, PUBLIC_TEST_DIR, TRAIN_DIR
from .dataset import HandWritttenDataset, collate_fn

def get_data(
        batch_size: int = 64,
        seed: int = 42,
        args=None
    ):
    """
    Get the train, validation and test data loaders

    Arguments:
    ----------

    batch_size: int (default: 64)
        The batch size to use for the data loaders

    seed: int (default: 42)
        The seed to use for the random number generator

    args:
        The arguments passed to the program
        
    Returns:
    --------
        train_loader, val_loader, test_loader
    """
    pl.seed_everything(seed)
    np.random.seed(seed)
    
    if args.resize == 1:
        train_transform = transforms.Compose([
            # Gaussian Noise
            transforms.GaussianBlur(3),
            # Color Jitter
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            # Random Rotation
            transforms.RandomRotation(15),
            # Random Cutout
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
            # Radom Grayscale
            transforms.RandomGrayscale(p=0.2),
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5818, 0.5700, 0.5632], 
                [0.1417, 0.1431, 0.1367]
            )
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5818, 0.5700, 0.5632], 
                [0.1417, 0.1431, 0.1367]
            )
        ])
    else:
        train_transform = transforms.Compose([
            # Gaussian Noise
            transforms.GaussianBlur(3),
            # Color Jitter
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            # Random Rotation
            transforms.RandomRotation(15),
            # Random Cutout
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
            # Radom Grayscale
            transforms.RandomGrayscale(p=0.2),
            FixedHeightResize(args.height), 
            FixedWidthPad(args.width),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5818, 0.5700, 0.5632], 
                [0.1417, 0.1431, 0.1367]
            )
        ])
        test_transform = transforms.Compose([
            FixedHeightResize(args.height), 
            FixedWidthPad(args.width),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5818, 0.5700, 0.5632], 
                [0.1417, 0.1431, 0.1367]
            )
        ])

    train_dataset = HandWritttenDataset(
        TRAIN_DIR, LABEL_FILE,
        name='train', transform=train_transform
    )
    val_dataset = HandWritttenDataset(
        TRAIN_DIR, LABEL_FILE,
        name='train', transform=test_transform
    )
    test_dataset = HandWritttenDataset(
        PUBLIC_TEST_DIR,
        name='public_test', transform=test_transform
    )

    form_inds = np.arange(0, 51000)
    wild_inds = np.arange(51000, 99000)
    gan_inds = np.arange(99000, 103000)
    np.random.shuffle(form_inds)
    np.random.shuffle(wild_inds)
    # Use GAN data only for training
    train_inds = np.concatenate([
        form_inds[5100:],
        wild_inds[4800:],
        gan_inds
    ])
    val_inds = np.concatenate([
        form_inds[:5100],
        wild_inds[:4800]
    ])
    train_set = Subset(train_dataset, train_inds)
    val_set = Subset(val_dataset, val_inds)

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, drop_last=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


class FixedHeightResize:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        w, h = img.size
        aspect_ratio = float(h) / float(w)
        new_w = math.ceil(self.size / aspect_ratio)
        return F.resize(img, (self.size, new_w))
    
# Pad to fixed width
class FixedWidthPad:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        w, h = img.size
        pad = self.size - w
        pad_left = pad // 2
        pad_right = pad - pad_left
        return F.pad(img, (pad_left, 0, pad_right, 0), 0, 'constant')