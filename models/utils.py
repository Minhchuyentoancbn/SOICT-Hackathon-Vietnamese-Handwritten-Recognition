def rule(args):
    train_steps = 93100 // args.batch_size * args.epochs
    WARM_UP_STEP = train_steps / 2
    def lr_update_rule(step):
        if step < WARM_UP_STEP:
            return  1.0
        else:
            return (train_steps - step)/(train_steps - WARM_UP_STEP)
    return lr_update_rule