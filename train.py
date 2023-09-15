import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
from pytorch_lightning.callbacks import StochasticWeightAveraging

import numpy as np
import pandas as pd
import pickle
import os
import math
from dataset import HandWrittenDataset, Align, collate_fn, DataAugment, OtsuGrayscale
from config import LABEL_FILE, PUBLIC_TEST_DIR, TRAIN_DIR, SYNTH_LABEL_FILE, SYNTH_TRAIN_DIR
from utils import AttnLabelConverter, CTCLabelConverter, TokenLabelConverter, SRNConverter, ParseqConverter, make_submission
from baseline import Model, LightningModel
from test import predict
from models.parseq import PARSeq


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
    
    data_augment = DataAugment(prob=args.augment_prob)

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
    
    train_transform = transforms.Compose([
        # Random Rotation
        transforms.RandomRotation(15, fill=(0.5818 * 255, 0.5700 * 255, 0.5632 * 255)),
        data_augment,
        grayscale,
        align
    ])
    test_transform = transforms.Compose([
        grayscale,
        align
    ])

    # Get the datasets
    train_dataset = HandWrittenDataset(
        TRAIN_DIR, LABEL_FILE,
        name='train_img', transform=train_transform
    )
    val_dataset = HandWrittenDataset(
        TRAIN_DIR, LABEL_FILE,
        name='train_img', transform=test_transform
    )
    test_dataset = HandWrittenDataset(
        PUBLIC_TEST_DIR,
        name='public_test_img', transform=test_transform
    )

    # Split the training data into train and validation
    if args.train:
        # Split the training data into train and validation
        if os.path.exists('train_inds.pkl') and os.path.exists('val_inds.pkl'):
            # Load the indices
            with open('train_inds.pkl', 'rb') as f:
                train_inds = pickle.load(f)
            with open('val_inds.pkl', 'rb') as f:
                val_inds = pickle.load(f)
        else:
            # NOTE: Use GAN data only for training
            # Check whether train_inds and val_inds files are present
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
            # Save the indices for later ensemble
            with open('train_inds.pkl', 'wb') as f:
                pickle.dump(train_inds, f)
            with open('val_inds.pkl', 'wb') as f:
                pickle.dump(val_inds, f)

        # Sample from the training data
        if args.num_samples > 0:
            train_inds = np.random.choice(train_inds, args.num_samples, replace=False)

        train_set = Subset(train_dataset, train_inds)
        val_set = Subset(val_dataset, val_inds)
    else:  # Use all training data for training
        print('Using all training data for training')
        train_set = train_dataset
        if args.num_samples > 0:
            train_set = Subset(train_dataset, np.random.choice(len(train_dataset), args.num_samples, replace=False))
        val_set = None

    # Add SynthText data
    if args.synth:
        synth_dataset = HandWrittenDataset(
            SYNTH_TRAIN_DIR, SYNTH_LABEL_FILE,
            name='gen_image', transform=test_transform
        )
        print('Using SynthText data for training')

        # Filter out the long labels
        synth_label_file = pd.read_csv('data/annotation2.txt', sep='\t', header=None, na_filter=False)
        synth_inds =  np.arange(len(synth_dataset))[(synth_label_file[1].str.len() < args.max_len)]
        if args.num_synth > 0:
            synth_inds = np.random.choice(synth_inds, args.num_synth, replace=False) # np.arange(args.num_synth)
        
        synth_set = Subset(synth_dataset, synth_inds)
        train_set = ConcatDataset([train_set, synth_set])

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
    if args.transformer:
        converter = TokenLabelConverter(args.max_len)
    elif args.prediction == 'ctc':
        converter = CTCLabelConverter()
    elif args.prediction == 'attention':
        converter = AttnLabelConverter()
    elif args.prediction == 'srn':
        converter = SRNConverter()
    elif args.prediction == 'parseq':
        converter = ParseqConverter()
        
    NUM_CLASSES = converter.num_classes
    
    if args.grayscale:
        input_channel = 1
    else:
        input_channel = 3

    # Get the model
    if args.prediction == 'parseq': # Use PARSeq
        if args.parseq_model == 'small' or args.parseq_model == 'small_pretrained':
            embed_dim = 384
            num_heads = 6
        elif args.parseq_model == 'base' or args.parseq_model == 'base_pretrained':
            embed_dim = 768
            num_heads = 12

        model = PARSeq(
            args.max_len, NUM_CLASSES, converter.pad_id, converter.bos_id, converter.eos_id, 
            (args.height, args.width), stn_on=args.stn_on, seed=args.seed, img_channel=input_channel,
            embed_dim=embed_dim, enc_num_heads=num_heads, patch_size=args.patch_size, refine_iters=args.refine_iters,
            pretrained=args.parseq_pretrained, transformer=args.parseq_use_transformer, model_name=args.parseq_model
        )
    else:  
        model = Model(
            input_channel, args.height, args.width, NUM_CLASSES,
            args.stn_on, args.feature_extractor, args.prediction,
            dropout=args.dropout, max_len=args.max_len, 
            transformer=args.transformer, transformer_model=args.transformer_model,
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
    preds, img_names, confidences = predict(pl_model.model, test_loader, converter, args.prediction, args.max_len, args.transformer)
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
    if args.swa:
        # Use stochastic weight averaging
        swa_epoch_start = 0.75
        swa_lr = args.lr * get_swa_lr_factor(0.075, swa_epoch_start)
        swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)
        callbacks = [swa, ]
    else:
        callbacks = None

    if args.train:
        # train model
        trainer = pl.Trainer(
            default_root_dir=f'checkpoints/{args.model_name}/',
            max_epochs=args.epochs,
            val_check_interval=args.val_check_interval,
            callbacks=callbacks
        )
        trainer.fit(
            model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
    else:
        # train model
        trainer = pl.Trainer(
            default_root_dir=f'checkpoints/{args.model_name}/',
            max_epochs=args.epochs,
            callbacks=callbacks
        )
        trainer.fit(
            model=pl_model, train_dataloaders=train_loader
        )

    return pl_model


# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000  # Can be anything. We use 1000 for convenience.
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)