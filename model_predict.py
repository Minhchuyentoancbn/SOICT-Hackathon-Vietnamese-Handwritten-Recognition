import sys
import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from PIL import Image
from tools import load_model
from test import predict
from config import PRIVATE_TEST_DIR, MODEL_PATH
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from dataset import Align, OtsuGrayscale


class PrivateDataset(Dataset):
    """Hand Writtten dataset."""

    def __init__(self, root_dir: str, label_file: str = None, name: str = 'private_test', transform=None, max_len: int = 25):
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
        image_idx = []
        for file in os.listdir(self.root_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_name = file.split('.')[0]
                image_idx.append(int(img_name.split('_')[-1]))
        
        self.image_idx = sorted(image_idx)


    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Read image
        real_idx = self.image_idx[idx]
        try:
            img_name = f'{self.name}_{real_idx}.jpg'
            image = Image.open(os.path.join(self.root_dir, img_name))
        except:
            img_name = f'{self.name}_{real_idx}.png'
            image = Image.open(os.path.join(self.root_dir, img_name))
        # Transform image
        if self.transform:
            image = self.transform(image)
        # Read label
        if self.labels is not None:
            label = self.labels.iloc[real_idx, 1]
            return image, label
        else:
            return image, img_name


def get_test_data(
        path,
        name='private_test',
        batch_size: int = 64,
        seed: int = 42,
        args=None
    ):
    """
    Get the train, validation and test data loaders

    Arguments:
    ----------
    path: str
        The path to the data

    name: str
        The name of the dataset
    
    batch_size: int (default: 64)
        The batch size to use for the data loaders

    seed: int (default: 42)
        The seed used to spli the data

    args:
        The arguments passed to the program
        
    Returns:
    --------
        train_loader, val_loader, test_loader, train_set, val_set, test_set
    """
    pl.seed_everything(seed)
    np.random.seed(seed)

    # Get the transforms
    if args.grayscale:
        if args.otsu:
            grayscale = OtsuGrayscale()
        else:
            grayscale = transforms.Grayscale()
        align = Align(1, args.height, args.width, args.keep_ratio_with_pad, args.transformer)  # 1 channel for grayscale
    else:
        grayscale = transforms.Compose([])  # Do nothing
        align = Align(3, args.height, args.width, args.keep_ratio_with_pad, args.transformer)
    
    test_transform = transforms.Compose([
        grayscale,
        align
    ])

    test_dataset = PrivateDataset(
        path,
        name=name, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=args.pin_memory
    )

    return test_loader,  test_dataset


def get_rotate_test_data(
        path,
        name='private_test',
        batch_size: int = 64,
        seed: int = 42,
        args=None,
        degree=0
    ):
    """
    Get the train, validation and test data loaders

    Arguments:
    ----------
    path: str
        The path to the data

    name: str
        The name of the dataset
    
    batch_size: int (default: 64)
        The batch size to use for the data loaders

    seed: int (default: 42)
        The seed used to spli the data

    args:
        The arguments passed to the program
        
    Returns:
    --------
        train_loader, val_loader, test_loader, train_set, val_set, test_set
    """
    pl.seed_everything(seed)
    np.random.seed(seed)

    # Get the transforms
    if args.grayscale:
        if args.otsu:
            grayscale = OtsuGrayscale()
        else:
            grayscale = transforms.Grayscale()
        align = Align(1, args.height, args.width, args.keep_ratio_with_pad, args.transformer)  # 1 channel for grayscale
    else:
        grayscale = transforms.Compose([])  # Do nothing
        align = Align(3, args.height, args.width, args.keep_ratio_with_pad, args.transformer)
    
    test_transform = transforms.Compose([
        RotationTransform(degree),
        grayscale,
        align
    ])

    test_dataset = PrivateDataset(
        path,
        name=name, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=args.pin_memory
    )

    return test_loader


if __name__ == '__main__':

    # Predict no rotation
    model_name = sys.argv[1]
    # Set seed
    pl.seed_everything(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    model, converter, args = load_model(model_name, MODEL_PATH)

    # Get the data
    test_loader, test_set = get_test_data(PRIVATE_TEST_DIR, batch_size=args.batch_size, seed=args.seed, args=args)

    # Make submission
    preds, img_names, confidences = predict(model, test_loader, converter, args.prediction, args.max_len, args.transformer)

    # Save the confidence for later ensemble
    df = pd.DataFrame({'img_name': img_names, 'confidence': confidences, 'pred': preds})
    df.to_csv(f'ensemble/private_test/{args.model_name}.csv', index=False)

    test_frame0 = df

    test_frames = [df, ]
    for i, degree in enumerate([-15, 15]):
        # Get the data
        test_loader = get_test_data(PRIVATE_TEST_DIR, batch_size=args.batch_size, seed=args.seed, args=args, degree=degree)
        preds, img_names, confidences = predict(model, test_loader, converter, args.prediction, args.max_len, args.transformer)
        test_frame = pd.DataFrame({'img_name': img_names, 'pred': preds, 'confidence': confidences})
        test_frames.append(test_frame)

    test_confidences = np.array([test_frame['confidence'] for test_frame in test_frames]).T
    test_predictions = np.array([test_frame['pred'] for test_frame in test_frames]).T
    test_idx = np.argmax(test_confidences, axis=1)
    test_pred = [test_predictions[i, test_idx[i]] for i in range(len(test_idx))]
    test_img_names = test_frames[0]['img_name']
    test_confidences = np.max(test_confidences, axis=1)

    test = pd.DataFrame({'img_name': test_img_names, 'pred': test_pred, 'confidence': test_confidences})
    test.to_csv(f'ensemble/private_test/{name}_aug15.csv', index=False)


    test_frames = [df, ]
    for i, degree in enumerate([-10, 10]):
        # Get the data
        test_loader = get_test_data(PRIVATE_TEST_DIR, batch_size=args.batch_size, seed=args.seed, args=args, degree=degree)
        preds, img_names, confidences = predict(model, test_loader, converter, args.prediction, args.max_len, args.transformer)
        test_frame = pd.DataFrame({'img_name': img_names, 'pred': preds, 'confidence': confidences})
        test_frames.append(test_frame)

    test_confidences = np.array([test_frame['confidence'] for test_frame in test_frames]).T
    test_predictions = np.array([test_frame['pred'] for test_frame in test_frames]).T
    test_idx = np.argmax(test_confidences, axis=1)
    test_pred = [test_predictions[i, test_idx[i]] for i in range(len(test_idx))]
    test_img_names = test_frames[0]['img_name']
    test_confidences = np.max(test_confidences, axis=1)

    test = pd.DataFrame({'img_name': test_img_names, 'pred': test_pred, 'confidence': test_confidences})
    test.to_csv(f'ensemble/private_test/{name}_aug10.csv', index=False)
    