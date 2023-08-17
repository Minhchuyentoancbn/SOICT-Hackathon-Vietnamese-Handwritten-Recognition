def rule(args):
    train_steps = 93100 // args.batch_size * args.epochs
    WARM_UP_STEP = args.warmup_steps
    def lr_update_rule(step):
        if step < 7000:
            return  1.0
        else:
            return (train_steps - step)/(train_steps - WARM_UP_STEP)
    return lr_update_rule