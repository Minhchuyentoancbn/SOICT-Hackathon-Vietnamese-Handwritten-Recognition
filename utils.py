import torch
import pickle
import pandas as pd
from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self):
        # character (str): set of the possible characters.
        character = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        dict_character = list(character)
        self.num_classes = len(dict_character) + 1  # +1 for [CTCblank] token for CTCLoss (index 0)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1
        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)


    def encode(self, text, batch_max_length=25):
        """
        Convert text-label into text-index.
        
        Arguments:
        ----------
        
        text: 
            text labels of each image. [batch_size]

        batch_max_length: 
            max length of text label in the batch. 25 by default

        Returns:
        -------
        text: 
            text index for CTCLoss. [batch_size, batch_max_length]
        
        length: 
            length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label."""
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)
            texts.append(text)
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self):
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        character = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character
        self.num_classes = len(self.character)

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i


    def encode(self, text, batch_max_length=25):
        """ 
        Convert text-label into text-index.

        Arguments:
        ----------
        text: 
            text labels of each image. [batch_size]
            
        batch_max_length: 
            max length of text label in the batch. 25 by default

        Returns:
        -------
        text : 
            the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
            text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            
        length : 
            the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """

        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.

        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))


    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
    

class TokenLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        character = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        self.SPACE = '[s]'
        self.GO = '[GO]'

        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(character)

        self.dict = {word: i for i, word in enumerate(self.character)}

    def encode(self, text, batch_max_length=25):
        """ 
        Convert text-label into text-index.
        """
        length = [len(s) + len(self.list_token) for s in text]  # +2 for [GO] and [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), batch_max_length + len(self.list_token)).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)  # batch_text[:, 0] = [GO] token
        return batch_text.to(device)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        # count = v.data.numel()
        # v = v.data.sum()
        self.n_count += 1#count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def load_char_dict():
    # Load char_dict from file
    with open('data/char_dict.pickle', 'rb') as handle:
        char_dict = pickle.load(handle)
    return char_dict


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    return device


def make_submission(preds, img_names, name):
    """
    Save predictions to file for submission.

    Arguments:
    ----------
    preds: list
        List of predicted labels.

    img_names: list
        List of image names.

    name: str
        Name of the model.
    """
    for i in range(len(preds)):
        # # Remove [s] and [GO] tokens
        # preds[i] = preds[i].replace('[s]', '')
        # preds[i] = preds[i].replace('[GO]', '')
        if len(preds[i]) == 0:
            preds[i] = 'ơ'

    df = pd.DataFrame({'file_name': img_names, 'pred': preds})
    df.to_csv(f'predictions/{name}.txt', index=False, header=False, sep='\t')


def parse_arguments(argv):
    parser = ArgumentParser()
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed, default: 42')
    parser.add_argument('--val_check_interval', type=float, default=0.5, help='Validation check interval per epoch, default: 0.5')
    parser.add_argument('--model_name', type=str, help='Name of the model to train')

    # Optimization hyperparameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs, default: 5')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size, default: 64')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate, default: 1e-3')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum, default: 0.9')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer, default: adam, options: adam, sgd, adamw, adadelta')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay, default: 0.0')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout after image feature extractor, default: 0.0')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing for CrossEntropyLoss, default: 0.0')

    # Learning rate scheduler
    parser.add_argument('--scheduler', type=int, default=0, help='Whether to use learning rate scheduler or not, default: 0 (not use scheduler)')
    parser.add_argument('--decay_epochs', type=int, default=0, help='Decay learning rate after this number of epochs, default: 0')

    # Data processing
    parser.add_argument('--train', type=int, default=1, help='Whether to use all training data or not, default: 1 (not use all training data)')
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to train on, default: 0 (all samples)')
    parser.add_argument('--keep_ratio_with_pad', type=int, default=1, help='Whether to keep ratio for image resize, default: 1 (keep ratio)')
    parser.add_argument('--height', type=int, default=32, help='Height of the input image, default: 32')
    parser.add_argument('--width', type=int, default=128, help='Width of the input image, default: 128')
    parser.add_argument('--grayscale', type=int, default=0, help='Convert image to grayscale which then has 1 channel, default: 0')

    # Model hyperparameters
    parser.add_argument('--feature_extractor', type=str, default='resnet', help='Feature extractor, default: resnet, options: resnet, vgg')
    parser.add_argument('--stn_on', type=int, default=0, help='Whether to use STN or not, default: 0 (not use STN)')
    parser.add_argument('--prediction', type=str, default='ctc', help='Prediction method, default: ctc, options: ctc, attention')
    parser.add_argument('--max_len', type=int, default=25, help='Max length of the predicted text, default: 25')

    return parser.parse_args(argv)