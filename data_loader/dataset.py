import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

class HandWritttenDataset(Dataset):
    """Hand Writtten dataset."""

    CHARS = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir: str, label_file: str = None, name: str = 'train', transform=None):
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
        """
        if label_file is not None:
            self.labels = pd.read_csv(label_file, sep='\t', header=None, encoding='utf-8', na_filter=False)
        else:
            self.labels = None
        self.transform = transform
        self.root_dir = root_dir
        self.name = name

    def __len__(self):
        return len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            img_name = f'{self.name}_img_{idx}.jpg'
            image = Image.open(os.path.join(self.root_dir, img_name))
        except:
            img_name = f'{self.name}_img_{idx}.png'
            image = Image.open(os.path.join(self.root_dir, img_name))

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels.iloc[idx, 1]
            target = [self.CHAR2LABEL[char] for char in label]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image, img_name
        

def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths