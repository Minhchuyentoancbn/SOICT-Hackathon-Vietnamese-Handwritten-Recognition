def lr_update_rule(step):
    if step < 1500:
        return  min(1.0, step / 1500)
    elif step < 2000:
        return 1.0
    else:
        return 0.1