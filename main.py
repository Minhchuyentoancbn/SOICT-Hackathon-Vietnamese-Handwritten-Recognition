import torch
import pytorch_lightning as pl
import sys
import warnings
from train import train
from argparse import ArgumentParser


def parse_arguments(argv):
    parser = ArgumentParser()
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed, default: 42')
    parser.add_argument('--val_check_interval', type=float, default=0.5, help='Validation check interval per epoch, default: 0.5')
    parser.add_argument('--model_name', type=str, help='Name of the model to train')
    parser.add_argument('--save', type=int, default=0, help='Whether to save for training later or not, default: 0 (not save)')

    # Optimization hyperparameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs, default: 5')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size, default: 64')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate, default: 1e-3')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum, default: 0.9')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer, default: adam, options: adam, sgd, adamw, adadelta')
    parser.add_argument('--timm_optim', type=int, default=0, help='Whether to use timm optimizer or not, default: 0 (not use timm optimizer)')
    parser.add_argument('--clip_grad_val', type=float, default=5.0, help='Clip gradient value, default: 5.0')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay, default: 0.0')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout after image feature extractor, default: 0.0')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing for CrossEntropyLoss, default: 0.0')
    parser.add_argument('--focal_loss', type=int, default=0, help='Whether to use focal loss or not, default: 0 (not use focal loss)')
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0, help='Gamma for focal loss, default: 2.0')
    parser.add_argument('--focal_loss_alpha', type=float, default=0.5, help='Alpha for focal loss, default: 0.5')

    # Learning rate scheduler
    parser.add_argument('--scheduler', type=int, default=0, help='Whether to use learning rate scheduler or not, default: 0 (not use scheduler)')
    parser.add_argument('--decay_epochs', type=int, nargs='+', default=[1, ], help='Decay learning rate after this number of epochs, default: 0')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for learning rate scheduler, default: 0.1')
    parser.add_argument('--one_cycle', type=int, default=0, help='Whether to use one cycle policy or not, default: 0 (not use one cycle policy)')
    parser.add_argument('--swa', type=int, default=0, help='Whether to use stochastic weight averaging or not, default: 0 (not use swa)')

    # Data processing
    parser.add_argument('--train', type=int, default=1, help='Whether to use all training data or not, default: 1 (not use all training data)')
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to train on, default: 0 (all samples)')
    parser.add_argument('--keep_ratio_with_pad', type=int, default=0, help='Whether to keep ratio for image resize, default: 0 (keep ratio)')
    parser.add_argument('--height', type=int, default=32, help='Height of the input image, default: 32')
    parser.add_argument('--width', type=int, default=128, help='Width of the input image, default: 128')
    parser.add_argument('--grayscale', type=int, default=0, help='Convert image to grayscale which then has 1 channel, default: 0')
    parser.add_argument('--otsu', type=int, default=0, help='Whether to use Otsu thresholding or not, default: 0 (not use Otsu thresholding)')
    parser.add_argument('--augment_prob', type=float, default=0.2, help='Augmentation probability, default: 0.2')

    # SynthText
    parser.add_argument('--synth', type=int, default=0, help='Whether to use SynthText or not, default: 0 (not use SynthText)')
    parser.add_argument('--num_synth', type=int, default=0, help='Number of SynthText samples to train on, default: 0 (all samples)')

    # Model hyperparameters
    # VitSTR
    choices = ["vitstr_tiny_patch16_224", "vitstr_small_patch16_224", "vitstr_base_patch16_224", ]
    parser.add_argument('--transformer', type=int, default=0, help='Whether to use transformer or not, default: 0 (not use transformer)')
    parser.add_argument('--transformer_model', type=str, default=choices[0], help='VitSTR model, default: vitstr_tiny_patch16_224, options: vitstr_tiny_patch16_224, vitstr_small_patch16_224, vitstr_base_patch16_224')

    # Parseq
    parser.add_argument('--parseq_model', type=str, default='small', help='Parseq model, default: small, options: small, base')
    parser.add_argument('--parseq_pretrained', type=int, default=0, help='Whether to use pretrained Parseq or not, default: 0 (not use pretrained Parseq)')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[4, 8], help='Patch size for Parseq, default: [4, 8]')
    parser.add_argument('--refine_iters', type=int, default=1, help='Number of refinement iterations, default: 1')

    # Other models
    parser.add_argument('--feature_extractor', type=str, default='resnet', help='Feature extractor, default: resnet, options: resnet, vgg, densenet')
    parser.add_argument('--stn_on', type=int, default=0, help='Whether to use STN or not, default: 0 (not use STN)')
    parser.add_argument('--prediction', type=str, default='ctc', help='Prediction method, default: ctc, options: ctc, attention, srn, parseq')
    parser.add_argument('--max_len', type=int, default=25, help='Max length of the predicted text, default: 25')

    # Auxiliary loss
    parser.add_argument('--count_mark', type=int, default=0, help='Whether to count mark or not, default: 0 (not count mark)')
    parser.add_argument('--mark_alpha', type=float, default=1.0, help='Alpha for mark, default: 1.0')
    parser.add_argument('--count_case', type=int, default=0, help='Whether to count uppercase or not, default: 0 (not count mark)')
    parser.add_argument('--case_alpha', type=float, default=1.0, help='Alpha for uppercase, default: 1.0')
    parser.add_argument('--count_char', type=int, default=0, help='Whether to count number of characters or not, default: 0 (not count number of characters)')
    parser.add_argument('--char_alpha', type=float, default=1.0, help='Alpha for number of characters, default: 1.0')

    return parser.parse_args(argv)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_arguments(sys.argv[1:])

    # Set seed
    pl.seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train(args)
    