import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import numpy as np
import pandas as pd
from dataset import HandWrittenDataset, Align, collate_fn, DataAugment
from config import LABEL_FILE, PUBLIC_TEST_DIR, TRAIN_DIR
from utils import AttnLabelConverter, CTCLabelConverter, make_submission
from baseline import Model, LightningModel
from test import predict


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
        The seed used to spli the data

    args:
        The arguments passed to the program
        
    Returns:
    --------
        train_loader, val_loader, test_loader, train_set, val_set, test_set
    """
    pl.seed_everything(seed)
    np.random.seed(seed)
    
    data_augment = DataAugment(prob=0.2)
    if args.grayscale:
        grayscale = transforms.Grayscale()
        align = Align(1, args.height, args.width, args.keep_ratio_with_pad, args.transformer)  # 1 channel for grayscale
    else:
        grayscale = transforms.Compose([])  # Do nothing
        align = Align(3, args.height, args.width, args.keep_ratio_with_pad, args.transformer)
    
    train_transform = transforms.Compose([
        data_augment,
        grayscale,
        align
    ])
    test_transform = transforms.Compose([
        grayscale,
        align
    ])

    train_dataset = HandWrittenDataset(
        TRAIN_DIR, LABEL_FILE,
        name='train', transform=train_transform
    )
    val_dataset = HandWrittenDataset(
        TRAIN_DIR, LABEL_FILE,
        name='train', transform=test_transform
    )
    test_dataset = HandWrittenDataset(
        PUBLIC_TEST_DIR,
        name='public_test', transform=test_transform
    )

    if args.train:
        # Split the training data into train and validation
        # NOTE: Use GAN data only for training
        form_inds = np.arange(0, 51000)
        wild_inds = np.arange(51000, 99000)
        gan_inds = np.arange(99000, 103000)
        np.random.shuffle(form_inds)
        np.random.shuffle(wild_inds)
        train_inds = np.concatenate([
            form_inds[5100:],
            wild_inds[4800:],
            gan_inds
        ])
        val_inds = np.concatenate([
            form_inds[:5100],
            wild_inds[:4800]
        ])
        # Sample from the training data
        if args.num_samples > 0:
            train_inds = np.random.choice(train_inds, args.num_samples, replace=False)

        train_set = Subset(train_dataset, train_inds)
        val_set = Subset(val_dataset, val_inds)
    else:
        print('Using all training data for training')
        train_set = train_dataset
        if args.num_samples > 0:
            train_set = Subset(train_dataset, np.random.choice(len(train_dataset), args.num_samples, replace=False))
        val_set = None

    # Set up the data loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, drop_last=True, collate_fn=collate_fn,
        pin_memory=True, num_workers=2
    )
    if args.train:
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
            pin_memory=True, num_workers=2
        )
    else:
        val_loader = None

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def train(args):
    # Get the data
    train_loader, val_loader, test_loader, train_set, val_set, test_set = get_data(args.batch_size, args.seed, args)

    # Get the converter
    if args.prediction == 'ctc':
        converter = CTCLabelConverter()
    elif args.prediction == 'attention':
        converter = AttnLabelConverter()

    NUM_CLASSES = converter.num_classes
    if args.grayscale:
        input_channel = 1
    else:
        input_channel = 3

    # Get the model
    model = Model(
        input_channel, args.height, args.width, NUM_CLASSES,
        args.stn_on, args.feature_extractor, args.prediction,
        dropout=args.dropout, max_len=args.max_len
    )
    pl_model = LightningModel(model, converter, args)

    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     mode='min', patience=10, verbose=True
    # )

    # train model
    pl_model = train_model(pl_model, train_loader, val_loader, args)

    # Save models
    torch.save(pl_model.model.state_dict(), f'saved_models/{args.model_name}.pt')

    print('Predicting...')
    print('-' * 50)

    # Make submission
    preds, img_names, confidences = predict(pl_model.model, test_loader, converter, args.prediction, args.max_len)
    make_submission(preds, img_names, args.model_name)

    # Save the confidence for later ensemble
    df = pd.DataFrame({'img_name': img_names, 'confidence': confidences, 'pred': preds})
    df.to_csv(f'predictions/{args.model_name}.csv', index=False)


def train_model(pl_model, train_loader, val_loader, args):
    """
    Train the model in pytorch lightning

    Arguments:
    ----------
    pl_model: pl.LightningModule
        The model to train

    train_loader: torch.utils.data.DataLoader
        The dataloader of the training set

    val_loader: torch.utils.data.DataLoader
        The dataloader of the validation set

    args: argparse.Namespace
        The arguments of the program
    """
    # train model
    if args.train:
        # train model
        trainer = pl.Trainer(
            default_root_dir=f'checkpoints/{args.model_name}/',
            max_epochs=args.epochs,
            val_check_interval=args.val_check_interval,
        )
        trainer.fit(
            model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
    else:
        # train model
        trainer = pl.Trainer(
            default_root_dir=f'checkpoints/{args.model_name}/',
            max_epochs=args.epochs,
        )
        trainer.fit(
            model=pl_model, train_dataloaders=train_loader
        )

    return pl_model