import torch
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import EarlyStopping

from models.crnn import CRNN
from models.cnnctc import CNNCTC
from models.ctc_baseline import CTCBaseline
from models.safl import SAFL
from data_loader import get_data, HandWritttenDataset
from predict import predict_ctc, predict_safl
from utils import initilize_parameters, make_submission

def ctc(args):
    # Get the data
    train_loader, val_loader, test_loader, _, _, _ = get_data(args.batch_size, args.seed, args)
    pl.seed_everything(args.seed)
    NUM_CLASSES = len(HandWritttenDataset.CHARS) + 1

    # model
    if args.model_name == 'crnn':
        model = CRNN(3, args.height, args.width, NUM_CLASSES, dropout=args.dropout)
        model = initilize_parameters(model)
    elif args.model_name == 'cnnctc':
        model = CNNCTC(NUM_CLASSES)
    pl_model = CTCBaseline(model, args)

    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     mode='min', patience=10, verbose=True
    # )

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

    # Save models
    torch.save(pl_model.model.state_dict(), f'saved_models/{args.model_name}.pt')

    print('Predicting...')
    print('-' * 50)

    # Make submission
    label2char = HandWritttenDataset.LABEL2CHAR
    preds, img_names = predict_ctc(pl_model, test_loader, label2char, args.decode_method, args.beam_size)
    make_submission(preds, img_names, args)


def safl(args):
    # Get the data
    train_loader, val_loader, test_loader, train_set, val_set, test_set = get_data(args.batch_size, args.seed, args)
    pl.seed_everything(args.seed)
    NUM_CLASSES = train_set.rec_num_classes
    eos = train_set.char2id[train_set.EOS]
    pad = train_set.char2id[train_set.PADDING]
    model = SAFL(NUM_CLASSES, eos, args=args, stn_on=args.stn_on)

    # train model
    if args.train:
        # train model
        trainer = pl.Trainer(
            default_root_dir=f'checkpoints/{args.model_name}/',
            max_epochs=args.epochs,
            val_check_interval=args.val_check_interval,
        )
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
    else:
        # train model
        trainer = pl.Trainer(
            default_root_dir=f'checkpoints/{args.model_name}/',
            max_epochs=args.epochs,
        )
        trainer.fit(
            model=model, train_dataloaders=train_loader
        )

    # Save models
    torch.save(model.state_dict(), f'saved_models/{args.model_name}.pt')

    print('Predicting...')
    print('-' * 50)

    # Make submission
    label2char = train_set.id2char
    preds, img_names = predict_safl(model, test_loader, label2char, eos, pad)
    make_submission(preds, img_names, args)