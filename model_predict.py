import sys
import pandas as pd
import torch
import pytorch_lightning as pl

from tools import load_model, get_test_data
from test import predict
from config import PRIVATE_TEST_DIR


if __name__ == '__main__':
    model_name = sys.argv[1]
    # Set seed
    pl.seed_everything(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    model, converter, args = load_model(model_name)

    # Get the data
    test_loader, test_set = get_test_data(PRIVATE_TEST_DIR, batch_size=args.batch_size, seed=args.seed, args=args)

    # Make submission
    preds, img_names, confidences = predict(model, test_loader, converter, args.prediction, args.max_len, args.transformer)

    # Save the confidence for later ensemble
    df = pd.DataFrame({'img_name': img_names, 'confidence': confidences, 'pred': preds})
    df.to_csv(f'ensemble/private_test/{args.model_name}.csv', index=False)
