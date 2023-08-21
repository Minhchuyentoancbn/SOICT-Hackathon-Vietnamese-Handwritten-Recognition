from torch.nn import init
from argparse import ArgumentParser

def rule(args):
    if args.train:
        dset_size = 93100
    else:
        dset_size = 103000
    if args.num_samples > 0:
        dset_size = args.num_samples
    train_steps = dset_size // args.batch_size * args.epochs
    WARM_UP_STEP = dset_size // args.batch_size * args.warmup_steps
    def lr_update_rule(step):
        if step < WARM_UP_STEP:
            return  1.0
        else:
            return 0.1 #(train_steps - step)/(train_steps - WARM_UP_STEP)
    return lr_update_rule


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
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)

    # Learning rate scheduler
    parser.add_argument('--warmup_steps', type=int, default=0)

    # Data processing
    parser.add_argument('--resize', type=int, default=1)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--grayscale', type=int, default=0)

    # CTC decode hyperparameters
    parser.add_argument('--decode_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--feature_extractor', type=str, default='vgg')

    # Encoder decoder hyperparameters
    parser.add_argument('--stn_on', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    return parser.parse_args(argv)


def initialize_weights(model):
    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue