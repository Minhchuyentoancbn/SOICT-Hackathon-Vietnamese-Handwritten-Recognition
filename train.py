import torch
import sys
import warnings
from train_utils import ctc, encoder_decoder
from models.utils import parse_arguments


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_arguments(sys.argv[1:])

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    if args.model_name in ['crnn', 'cnnctc']:
        ctc(args)
    elif args.model_name in ['aed']:
        encoder_decoder(args)