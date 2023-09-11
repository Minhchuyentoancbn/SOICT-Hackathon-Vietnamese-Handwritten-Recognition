import torch
import pytorch_lightning as pl
import sys
import warnings
from train import train
from utils import parse_arguments


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_arguments(sys.argv[1:])

    # Set seed
    pl.seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train(args)
    