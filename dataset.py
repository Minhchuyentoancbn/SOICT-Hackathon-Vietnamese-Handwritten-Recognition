import os
import math
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# from augmentation.geometry import Shrink
from augmentation.blur import GaussianBlur, MotionBlur
from augmentation.camera import Brightness, JpegCompression
from augmentation.process import Equalize, AutoContrast, Sharpness, Color


class HandWrittenDataset(Dataset):
    """Hand Writtten dataset."""

    def __init__(self, root_dir: str, label_file: str = None, name: str = 'train', transform=None, max_len: int = 25):
        """
        Arguments:
        ----------
        root_dir: str
            Directory with all the images.

        label_file: str
            Path to the label file.

        name: str
            Name of the dataset. Either 'train' or 'public_test'.

        transform: callable, optional
            Optional transform to be applied on a sample.

        max_len: int
            Maximum length of the label.
        """
        if label_file is not None:  # train
            self.labels = pd.read_csv(label_file, sep='\t', header=None, encoding='utf-8', na_filter=False)
        else:  # public_test
            self.labels = None
        self.transform = transform
        self.root_dir = root_dir
        self.name = name
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Read image
        try:
            img_name = f'{self.name}_{idx}.jpg'
            image = Image.open(os.path.join(self.root_dir, img_name))
        except:
            img_name = f'{self.name}_{idx}.png'
            image = Image.open(os.path.join(self.root_dir, img_name))
        # Transform image
        if self.transform:
            image = self.transform(image)
        # Read label
        if self.labels is not None:
            label = self.labels.iloc[idx, 1]
            return image, label
        else:
            return image, img_name
        

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    num_marks = count_denmark(labels)  # (N, 5), number of each type of diacritic mark
    num_uppercase = count_uppercase(labels)  # (N, 1), number of uppercase characters
    return images, labels, num_marks, num_uppercase
        

class ResizeNormalize(object):
    """
    Resize image to a fixed size, covert to torch.Tensor and normalize.
    """
    def __init__(self, size, interpolation=Image.BICUBIC, scale=True):
        """
        Arguments:
        ----------
        size: (int, int)
            Size of the resized image.

        interpolation: int
            Interpolation method.

        scale: bool
            Whether to scale the image or not.
        """
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.scale = scale

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        if self.scale:
            img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):
    """
    Convert image to torch.Tensor and normalize. Pad the width if needed.
    """
    def __init__(self, max_size, scale=True):
        """
        Arguments:
        ----------
        max_size: (c, h, w)
            Maximum size of the image.
        """
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.scale = scale

    def __call__(self, img):
        img = self.toTensor(img)
        if self.scale:
            img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class Align(object):
    """
    Resize image to a fixed size, and pad the width if needed.
    Convert image to torch.Tensor and normalize.
    """
    def __init__(self, imgC=3, imgH=32, imgW=128, keep_ratio_with_pad=False, transformer=False):
        """
        Arguments:
        ----------
        imgC: int
            Number of channels of the image.

        imgH: int
            Height of the resized image.

        imgW: int
            Width of the resized image.

        keep_ratio_with_pad: bool
            Whether to keep the ratio of the image or not.

        transform: bool
            Whether to use transformer or not.
        """
        self.imgC = imgC
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        if transformer:  # ViTSRT
            self.scale = False
        else:
            self.scale = True


    def __call__(self, img):
        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = self.imgC
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w), self.scale)

            w, h = img.size
            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = img.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_image = transform(resized_image)
        else:
            transform = ResizeNormalize((self.imgW, self.imgH), self.scale)
            resized_image = transform(img)
    
        return resized_image


class DataAugment(object):
    """
    Support with and without augmentations
    """
    def __init__(self, augment=True, prob=0.5, augs_num=2, augs_mag=None):
        """
        Arguments:
        ----------
        augment: bool
            Whether to use augmentations or not.

        prob: float
            Probability of using augmentations.

        augs_num: int
            Number of augmentations to use.

        augs_mag: int
            Magnitude of the augmentations.
        """
        self.augment = augment
        self.prob = prob
        self.augs_num = augs_num
        self.augs_mag = augs_mag

        if augment:
            self.process = [Equalize(), AutoContrast(), Sharpness(), Color()]
            self.camera = [Brightness(), JpegCompression()]
            self.blur = [GaussianBlur(),  MotionBlur()]
            # self.geometry = [Shrink()]

            self.augs = [self.process, self.camera, self.blur]


    def __call__(self, img):
        """
        Must call img.copy() if pattern, Rain or Shadow is used
        """
        random_prob = np.random.uniform(0,1)
        if self.augment and random_prob < self.prob:
            img = self.rand_aug(img)
        return img


    def rand_aug(self, img):
        aug_idx = np.random.choice(np.arange(len(self.augs)), self.augs_num, replace=False)
        for idx in aug_idx:
            aug = self.augs[idx]
            index = np.random.randint(0, len(aug))
            op = aug[index]
            mag = np.random.randint(0, 3) if self.augs_mag is None else self.augs_mag
            img = op(img, mag=mag)

        return img
    

class OtsuGrayscale(object):

    def __call__(self, img):
        a = np.array(img)
        #making 3 grayscale images 
        b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        # otsu thresholding
        thresh = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Morph open to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours and remove small noise
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 50:
                cv2.drawContours(opening, [c], -1, 0, -1)

        return Image.fromarray(opening)
    

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angles, fill=(0.5818 * 255, 0.5700 * 255, 0.5632 * 255))
    


def count_denmark(text):
    """
    Count the number for each type of denmark in a string.

    Arguments:
    ----------
    text: list(str)
        List of strings.

    Returns:
    --------
    marks: torch.Tensor(N, 5)
        The number of each type of denmark in the string.
    """
    sac = list('ÁÉÍÓÚÝáéíóúýẤấẮắẾếỐốỚớỨứ')
    huy = list('ÀÈÌÒÙỲàèìòùỳẦầẰằỀềỒồỜờỪừ')
    nga = list('ÃẼĨÕŨỸãẽĩõũỹẪẫẴẵỄễỖỗỠỡỮữ')
    nan = list('ẠẸỊỌỤỴạẹịọụỵẬậẶặỆệỘộỢợỰự')
    hoi = list('ẢẺỈỎỦỶảẻỉỏủỷẨẩẲẳỂểỔổỞởỬử')

    N = len(text)
    marks = torch.zeros(N, 5)
    for i, t in enumerate(text):
        for c in t:
            if c in sac:
                marks[i, 0] += 1
            elif c in huy:
                marks[i, 1] += 1
            elif c in nga:
                marks[i, 2] += 1
            elif c in nan:
                marks[i, 3] += 1
            elif c in hoi:
                marks[i, 4] += 1
    return marks


def count_uppercase(text):
    """
    Count the number of uppercase characters in a string.

    Arguments:
    ----------
    text: list(str)
        List of strings.

    Returns:
    --------
    uppercase: torch.Tensor(N, 1)
        The number of uppercase characters in the string.
    """
    N = len(text)
    uppercase = torch.zeros(N, 1)
    for i, t in enumerate(text):
        for c in t:
            if c.isupper():
                uppercase[i, 0] += 1
    return uppercase