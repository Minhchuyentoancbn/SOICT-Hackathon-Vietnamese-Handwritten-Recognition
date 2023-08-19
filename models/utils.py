from torch.nn import init

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