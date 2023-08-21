import torch
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import EarlyStopping

import pandas as pd
from models.crnn import CRNN
from models.cnnctc import CNNCTC
from models.ctc_baseline import CTCBaseline
from models.safl import SAFL
from models.aed import AttentionEncoderDecoder
from data_loader import get_data, HandWritttenDataset, AttnLabelConverter
from predict import predict_ctc, predict_aed
from utils import initilize_parameters, make_submission


def ctc(args):
    # Get the data
    train_loader, val_loader, test_loader, _, _, _ = get_data(args.batch_size, args.seed, args)
    pl.seed_everything(args.seed)
    NUM_CLASSES = len(HandWritttenDataset.CHARS) + 1

    # model
    if args.model_name == 'crnn':
        model = CRNN(3, args.height, args.width, NUM_CLASSES, dropout=args.dropout, feature_extractor=args.feature_extractor, stn_on=args.stn_on)
        if args.feature_extractor == 'vgg':
            model = initilize_parameters(model)
        # model = torch.compile(model)
    elif args.model_name == 'cnnctc':
        model = CNNCTC(NUM_CLASSES)
        
    pl_model = CTCBaseline(model, args)

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
    label2char = HandWritttenDataset.LABEL2CHAR
    preds, img_names = predict_ctc(pl_model, test_loader, label2char, args.decode_method, args.beam_size)
    make_submission(preds, img_names, args)


def encoder_decoder(args):
    # Get the data
    train_loader, val_loader, test_loader, _, _, _ = get_data(args.batch_size, args.seed, args)
    pl.seed_everything(args.seed)
    converter = AttnLabelConverter()
    NUM_CLASSES = converter.num_classes

    model = AttentionEncoderDecoder(args.height, args.width, NUM_CLASSES, converter, dropout=args.dropout, label_smoothing=args.label_smoothing)
    model = train_model(model, train_loader, val_loader, args)
    # Save models
    torch.save(model.state_dict(), f'saved_models/{args.model_name}.pt')

    print('Predicting...')
    print('-' * 50)

    # Make submission
    preds, img_names, confidences = predict_aed(model, test_loader, converter)
    make_submission(preds, img_names, args)

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