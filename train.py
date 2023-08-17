import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import sys
import warnings
import pandas as pd
from argparse import ArgumentParser
from models.crnn import CRNN
from models.cnnctc import CNNCTC
from models.ctc_baseline import CTCBaseline
from data_loader import get_data, HandWritttenDataset
from config import NUM_CLASSES
from predict import predict
from utils import initilize_parameters, make_submission


def parse_arguments(argv):
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_check_interval', type=float, default=0.25)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--train', type=bool, default=True)

    # Optimization hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.0)

    # Learning rate scheduler
    parser.add_argument('--warmup_steps', type=int, default=0)

    # Data processing
    parser.add_argument('--resize', type=int, default=1)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=256)

    # CTC decode hyperparameters
    parser.add_argument('--decode_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)

    return parser.parse_args(argv)


def ctc(args):
    train_loader, val_loader, test_loader = get_data(args.batch_size, args.seed, args)

    pl.seed_everything(args.seed)

    # model
    if args.model_name == 'crnn':
        model = CRNN(3, args.height, args.width, NUM_CLASSES, dropout=args.dropout)
        model = initilize_parameters(model)
    elif args.model_name == 'cnnctc':
        model = CNNCTC(NUM_CLASSES)

    pl_model = CTCBaseline(model, args)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min', patience=10, verbose=True
    )

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

    # Save models
    torch.save(pl_model.model.state_dict(), f'saved_models/{args.model_name}.pt')

    print('Predicting...')
    print('-' * 50)

    # Make submission
    label2char = HandWritttenDataset.LABEL2CHAR
    preds, img_names = predict(pl_model, test_loader, label2char, args.decode_method, args.beam_size)
    make_submission(preds, img_names, args)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_arguments(sys.argv[1:])

    if args.model_name in ['crnn', 'cnnctc']:
        ctc(args)