def rule(args):
    if args.train:
        dset_size = 93100
    else:
        dset_size = 103000
    train_steps = dset_size // args.batch_size * args.epochs
    WARM_UP_STEP = dset_size // args.batch_size * args.warmup_steps
    def lr_update_rule(step):
        if step < WARM_UP_STEP:
            return  1.0
        else:
            return (train_steps - step)/(train_steps - WARM_UP_STEP)
    return lr_update_rule