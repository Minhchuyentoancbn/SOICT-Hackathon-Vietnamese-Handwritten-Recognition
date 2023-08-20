import torch
import sys
import warnings
from argparse import ArgumentParser
from train_utils import ctc, safl


def parse_arguments(argv):
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_check_interval', type=float, default=0.25)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=0)

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
    parser.add_argument('--feature_extractor', type=str, default='vgg')

    # SAFL hyperparameters
    parser.add_argument('--stn_on', type=int, default=0)

    return parser.parse_args(argv)



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_arguments(sys.argv[1:])

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    if args.model_name in ['crnn', 'cnnctc']:
        ctc(args)
    elif args.model_name == 'safl':
        safl(args)