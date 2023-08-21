import pickle
import pandas as pd
import torch
import torch.nn as nn


def load_char_dict():
    # Load char_dict from file
    with open('data/char_dict.pickle', 'rb') as handle:
        char_dict = pickle.load(handle)
    return char_dict


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    return device


def initilize_parameters(model):
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0.0)
    return model


def make_submission(preds, img_names, args):
    for i in range(len(preds)):
        # Remove [s] and [GO] tokens
        preds[i] = preds[i].replace('[s]', '')
        preds[i] = preds[i].replace('[GO]', '')
        if len(preds[i]) == 0:
            preds[i] = 'ơ'

    df = pd.DataFrame({'file_name': img_names, 'pred': preds})
    df.to_csv(f'predictions/{args.model_name}.txt', index=False, header=False, sep='\t')