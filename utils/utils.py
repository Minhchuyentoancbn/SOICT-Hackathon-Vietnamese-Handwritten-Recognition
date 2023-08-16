import pickle
import torch

def load_char_dict():
    # Load char_dict from file
    with open('data/char_dict.pickle', 'rb') as handle:
        char_dict = pickle.load(handle)
    return char_dict

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    return device