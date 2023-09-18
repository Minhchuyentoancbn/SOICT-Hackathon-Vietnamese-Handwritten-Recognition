import torch
import torch.nn.functional as F
import pickle
import pandas as pd
import re
import string
import unicodedata
from baseline import Model
from models.parseq import PARSeq
from torchmetrics.text import CharErrorRate
from argparse import ArgumentParser


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_arguments(argv):
    parser = ArgumentParser()
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed, default: 42')
    parser.add_argument('--val_check_interval', type=float, default=0.5, help='Validation check interval per epoch, default: 0.5')
    parser.add_argument('--model_name', type=str, help='Name of the model to train')
    parser.add_argument('--save', type=int, default=0, help='Whether to save for training later or not, default: 0 (not save)')

    # Optimization hyperparameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs, default: 5')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size, default: 64')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate, default: 1e-3')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum, default: 0.9')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer, default: adam, options: adam, sgd, adamw, adadelta')
    parser.add_argument('--timm_optim', type=int, default=0, help='Whether to use timm optimizer or not, default: 0 (not use timm optimizer)')
    parser.add_argument('--clip_grad_val', type=float, default=5.0, help='Clip gradient value, default: 5.0')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay, default: 0.0')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout after image feature extractor, default: 0.0')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing for CrossEntropyLoss, default: 0.0')
    parser.add_argument('--focal_loss', type=int, default=0, help='Whether to use focal loss or not, default: 0 (not use focal loss)')
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0, help='Gamma for focal loss, default: 2.0')
    parser.add_argument('--focal_loss_alpha', type=float, default=0.5, help='Alpha for focal loss, default: 0.5')

    # Learning rate scheduler
    parser.add_argument('--scheduler', type=int, default=0, help='Whether to use learning rate scheduler or not, default: 0 (not use scheduler)')
    parser.add_argument('--decay_epochs', type=int, nargs='+', default=[1, ], help='Decay learning rate after this number of epochs, default: 0')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for learning rate scheduler, default: 0.1')
    parser.add_argument('--one_cycle', type=int, default=0, help='Whether to use one cycle policy or not, default: 0 (not use one cycle policy)')
    parser.add_argument('--swa', type=int, default=0, help='Whether to use stochastic weight averaging or not, default: 0 (not use swa)')

    # Data processing
    parser.add_argument('--train', type=int, default=1, help='Whether to use all training data or not, default: 1 (not use all training data)')
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to train on, default: 0 (all samples)')
    parser.add_argument('--keep_ratio_with_pad', type=int, default=0, help='Whether to keep ratio for image resize, default: 0 (keep ratio)')
    parser.add_argument('--height', type=int, default=32, help='Height of the input image, default: 32')
    parser.add_argument('--width', type=int, default=128, help='Width of the input image, default: 128')
    parser.add_argument('--grayscale', type=int, default=0, help='Convert image to grayscale which then has 1 channel, default: 0')
    parser.add_argument('--otsu', type=int, default=0, help='Whether to use Otsu thresholding or not, default: 0 (not use Otsu thresholding)')
    parser.add_argument('--augment_prob', type=float, default=0.2, help='Augmentation probability, default: 0.2')

    # SynthText
    parser.add_argument('--synth', type=int, default=0, help='Whether to use SynthText or not, default: 0 (not use SynthText)')
    parser.add_argument('--num_synth', type=int, default=0, help='Number of SynthText samples to train on, default: 0 (all samples)')

    # Model hyperparameters
    # VitSTR
    choices = ["vitstr_tiny_patch16_224", "vitstr_small_patch16_224", "vitstr_base_patch16_224", 'vitstr_small_pretrained_patch16_224', 'vitstr_base_pretrained_patch16_224',]
    parser.add_argument('--transformer', type=int, default=0, help='Whether to use transformer or not, default: 0 (not use transformer)')
    parser.add_argument('--transformer_model', type=str, default=choices[0], help='VitSTR model, default: vitstr_tiny_patch16_224, options: vitstr_tiny_patch16_224, vitstr_small_patch16_224, vitstr_base_patch16_224,vitstr_small_pretrained_patch16_224, vitstr_base_pretrained_patch16_224,')

    # Parseq
    parser.add_argument('--parseq_use_transformer', type=int, default=0, help='Whether to use transformer or not, default: 0 (not use transformer)')
    parser.add_argument('--parseq_model', type=str, default='small', help='Parseq model, default: small, options: small, base, small_pretrained, base_pretrained')
    parser.add_argument('--parseq_pretrained', type=int, default=0, help='Whether to use pretrained Parseq or not, default: 0 (not use pretrained Parseq)')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[4, 8], help='Patch size for Parseq, default: [4, 8]')
    parser.add_argument('--refine_iters', type=int, default=1, help='Number of refinement iterations, default: 1')

    # Other models
    parser.add_argument('--feature_extractor', type=str, default='resnet', help='Feature extractor, default: resnet, options: resnet, vgg, densenet')
    parser.add_argument('--stn_on', type=int, default=0, help='Whether to use STN or not, default: 0 (not use STN)')
    parser.add_argument('--prediction', type=str, default='ctc', help='Prediction method, default: ctc, options: ctc, attention, srn, parseq')
    parser.add_argument('--max_len', type=int, default=25, help='Max length of the predicted text, default: 25')

    # Auxiliary loss
    parser.add_argument('--count_mark', type=int, default=0, help='Whether to count mark or not, default: 0 (not count mark)')
    parser.add_argument('--mark_alpha', type=float, default=1.0, help='Alpha for mark, default: 1.0')
    parser.add_argument('--count_case', type=int, default=0, help='Whether to count uppercase or not, default: 0 (not count mark)')
    parser.add_argument('--case_alpha', type=float, default=1.0, help='Alpha for uppercase, default: 1.0')
    parser.add_argument('--count_char', type=int, default=0, help='Whether to count number of characters or not, default: 0 (not count number of characters)')
    parser.add_argument('--char_alpha', type=float, default=1.0, help='Alpha for number of characters, default: 1.0')

    return parser.parse_args(argv)



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

    def __init__(self, max_len=25):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        character = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        self.SPACE = '[s]'
        self.GO = '[GO]'

        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(character)
        self.num_classes = len(self.character)

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = max_len + len(self.list_token)  # +2 for [GO] and [s] at end of sentence.


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
    

class SRNConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self):
        # character (str): set of the possible characters.
        character = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ$#'
        self.character = list(character)
        self.num_classes = len(self.character)
        self.PAD = len(self.character) - 1
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(self.PAD).to(device)
        for i, t in enumerate(text):
            t = list(t + self.character[-2])
            text = [self.dict[char] for char in t]
            # t_mask = [1 for i in range(len(text) + 1)]
            batch_text[i][0:len(text)] = torch.LongTensor(text)  # batch_text[:, len_text+1] = [EOS] token
        return batch_text.to(device), torch.IntTensor(length).to(device)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            idx = text.find('$')
            texts.append(text[:idx])
        return texts
    

class ParseqConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, max_len=25):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        character = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        self.EOS = '[E]'
        self.BOS = '[B]'
        self.PAD = '[P]'

        self.character = [self.EOS] + list(character) + [self.BOS] + [self.PAD]
        self.num_classes = len(self.character) - 2  # Don't count [BOS] and [PAD] tokens

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.eos_id = self.dict[self.EOS]
        self.bos_id = self.dict[self.BOS]
        self.pad_id = self.dict[self.PAD]
        self.batch_max_length = max_len + 2  # +2 for [BOS] and [EOS] at end of sentence.


    def encode(self, text, batch_max_length=25):
        """ 
        Convert text-label into text-index.
        """
        length = [len(s) + 2 for s in text]  # +2 for [GO] and [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), batch_max_length + 2).fill_(self.pad_id)
        for i, t in enumerate(text):
            txt = [self.BOS] + list(t) + [self.EOS]
            txt = [self.dict[char] for char in txt]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)  # batch_text[:, 0] = [GO] token
        return batch_text.to(device)
    
    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            idx = text.find(self.EOS)
            texts.append(text[:idx])
        return texts
    


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


def read_scripts(name):
    """
    Read the script file and return the command.

    Arguments:
    ----------
    name: str
        Name of the script file. e.g. 'model1'

    Returns:
    --------
    command: str
        The command to run the script.
    """
    with open(f'scripts/{name}.sh') as f:
        lines = f.readlines()
        command = ' '.join([line.strip() for line in lines]).replace('\\ ', '')[16:]
        command = command.replace('--synth 1', '')  # For model that use synth data
        command = command.strip()
    return command


def load_model(name):
    """
    Load the model state dict from file.

    Arguments:
    ----------
    name: str
        Name of the model.

    Returns:
    --------
    model: torch.nn.Module
        The trained model.

    converter: Converter
        The converter used to convert between text-label and text-index.

    args: argparse.Namespace
        The arguments used to train the model.
    """
    command = read_scripts(name)
    args = parse_arguments(command.split())

    # Get the converter
    if args.transformer:
        converter = TokenLabelConverter(args.max_len)
    elif args.prediction == 'ctc':
        converter = CTCLabelConverter()
    elif args.prediction == 'attention':
        converter = AttnLabelConverter()
    elif args.prediction == 'srn':
        converter = SRNConverter()
    elif args.prediction == 'parseq':
        converter = ParseqConverter()
        
    NUM_CLASSES = converter.num_classes

    if args.grayscale:
        input_channel = 1
    else:
        input_channel = 3

    # Get the model
    if args.prediction == 'parseq':
        if args.parseq_model == 'small' or args.parseq_model == 'small_pretrained':
            embed_dim = 384
            num_heads = 6
        elif args.parseq_model == 'base' or args.parseq_model == 'base_pretrained':
            embed_dim = 768
            num_heads = 12

        model = PARSeq(
            args.max_len, NUM_CLASSES, converter.pad_id, converter.bos_id, converter.eos_id, 
            (args.height, args.width), stn_on=args.stn_on, seed=args.seed, img_channel=input_channel,
            embed_dim=embed_dim, enc_num_heads=num_heads, patch_size=args.patch_size, refine_iters=args.refine_iters,
            pretrained=args.parseq_pretrained, transformer=args.parseq_use_transformer, model_name=args.parseq_model,
        )
    else:
        model = Model(
            input_channel, args.height, args.width, NUM_CLASSES,
            args.stn_on, args.feature_extractor, args.prediction,
            dropout=args.dropout, max_len=args.max_len, 
            transformer=args.transformer, transformer_model=args.transformer_model,
        )

    model.load_state_dict(torch.load(f'saved_models/{name}.pt'))
    device = get_device()
    model.eval()
    model.to(device)

    return model, converter, args


def predict_train_valid(model, converter, data_loader, args):
    
    cer = CharErrorRate()
    device = get_device()
    preds_lst = []
    reals_lst = []
    img_lst = []
    cer_lst = []
    confidences = []
    max_length = args.max_len
    transformer = args.transformer

    with torch.no_grad():
        for batch in data_loader:
            # Prepare data
            images, labels, _, _ = batch
            images = images.to(device)
            batch_size = images.size(0)
            if not transformer:
                length_for_pred = torch.IntTensor([max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, max_length + 1).fill_(0).to(device)

            # Compute loss
            model.eval()
            if args.transformer:
                preds = model(images, seqlen=converter.batch_max_length, text=None)
                _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
                preds_index = preds_index.view(-1, converter.batch_max_length)
                length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
                preds_str = converter.decode(preds_index[:, 1:], length_for_pred)
            elif args.prediction == 'ctc':
                preds, _ = model(images, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2) # (B, T, C) -> (B, T), greedy decoding
                preds_str = converter.decode(preds_index, preds_size)
            elif args.prediction == 'attention':
                preds, _ = model(images, text_for_pred, is_train=False)
                _, preds_index = preds.max(2) # (B, T, C) -> (B, T), greedy decoding
                preds_str = converter.decode(preds_index, length_for_pred)
            elif args.prediction == 'srn':
                preds, _ = model(images, None)
                preds = preds[2]
                _, preds_index = preds.max(2) # (B, T, C) -> (B, T), greedy decoding
                preds_str = converter.decode(preds_index, length_for_pred)
            elif args.prediction == 'parseq':
                preds = model(images)
                _, preds_index = preds.max(2) # (B, T, C) -> (B, T), greedy decoding
                preds_str = converter.decode(preds_index, length_for_pred)

            # Compute confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            # Compute CER
            for i, (gt, pred, pred_max_prob) in enumerate(zip(labels, preds_str, preds_max_prob)):
                if args.transformer or args.prediction == 'attention':
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]
                elif args.prediction == 'srn' or args.prediction == 'parseq':
                    pred_EOS = len(pred)
                    pred_max_prob = pred_max_prob[:pred_EOS]

                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    confidences.append(confidence_score.item())
                except:
                    confidence_score = 0.0  # Case when pred_max_prob is empty
                    confidences.append(confidence_score)

                preds_lst.append(pred)
                reals_lst.append(gt)
                cer_lst.append(cer([pred], [gt]).item())
                img_lst.append(images[i].cpu().numpy().transpose(1, 2, 0))

    return preds_lst, reals_lst, cer_lst, confidences, img_lst


def normalize_diacritics(source, new_style=False, decomposed=False):
    tone = r'([\u0300\u0309\u0303\u0301\u0323])'
    combining_breve = r'\u0306'
    combining_circumflex_accent = r'\u0302'
    combining_horn = r'\u031B'
    diacritics = f'{combining_breve}{combining_circumflex_accent}{combining_horn}'
    result = unicodedata.normalize('NFD', source)
    # Put the tone on the second vowel
    result = re.sub(r'(?i){}([aeiouy{}]+)'.format(tone, diacritics), r'\2\1', result)
    # Put the tone on the vowel with a diacritic
    result = re.sub(r'(?i)(?<=[{}])(.){}'.format(diacritics, tone), r'\2\1', result)
    # For vowels that are not oa, oe, uy put the tone on the penultimate vowel
    result = re.sub(r'(?i)(?<=[ae])([iouy]){}'.format(tone), r'\2\1', result)
    result = re.sub(r'(?i)(?<=[oy])([iuy]){}'.format(tone), r'\2\1', result)
    result = re.sub(r'(?i)(?<!q)(u)([aeiou]){}'.format(tone), r'\1\3\2', result)
    result = re.sub(r'(?i)(?<!g)(i)([aeiouy]){}'.format(tone), r'\1\3\2', result)
    if not new_style:
        # Put tone in the symmetrical position
        result = re.sub(r'(?i)(?<!q)([ou])([aeoy]){}(?!\w)'.format(tone), r'\1\3\2', result)
    if decomposed:
        return unicodedata.normalize('NFD', result)
    return unicodedata.normalize('NFC', result)


# Taken from CLDR - Unicode Common Locale Data Repository
# http://demo.icu-project.org/icu-bin/locexp
ACCENTED_CHARACTERS = [
    'a',
    'à',
    'ả',
    'ã',
    'á',
    'ạ',
    'ă',
    'ằ',
    'ẳ',
    'ẵ',
    'ắ',
    'ặ',
    'â',
    'ầ',
    'ẩ',
    'ẫ',
    'ấ',
    'ậ',
    'b',
    'c',
    'd',
    'đ',
    'e',
    'è',
    'ẻ',
    'ẽ',
    'é',
    'ẹ',
    'ê',
    'ề',
    'ể',
    'ễ',
    'ế',
    'ệ',
    'f',
    'g',
    'h',
    'i',
    'ì',
    'ỉ',
    'ĩ',
    'í',
    'ị',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'ò',
    'ỏ',
    'õ',
    'ó',
    'ọ',
    'ô',
    'ồ',
    'ổ',
    'ỗ',
    'ố',
    'ộ',
    'ơ',
    'ờ',
    'ở',
    'ỡ',
    'ớ',
    'ợ',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'ù',
    'ủ',
    'ũ',
    'ú',
    'ụ',
    'ư',
    'ừ',
    'ử',
    'ữ',
    'ứ',
    'ự',
    'v',
    'w',
    'x',
    'y',
    'ỳ',
    'ỷ',
    'ỹ',
    'ý',
    'ỵ',
    'z',
]
VIETNAMESE_CHARACTERS = list(string.punctuation) + list(string.digits) + [x.upper() for x in ACCENTED_CHARACTERS] + ACCENTED_CHARACTERS


class VietnameseOrderDict(dict):
    def __init__(self):
        for i, char in enumerate(VIETNAMESE_CHARACTERS):
            self[char] = i

    def __getitem__(self, k):
        # Any characters that is not in the dictinary has its index shifted out by the number of keys in the dictionary
        # to preserve relative order and give precedence to characters used in Vietnamese
        if k not in self:
            return ord(k) + len(self)
        return super().__getitem__(k)


vietnamese_order_dict = VietnameseOrderDict()


def vietnamese_sort_key(word):
    word = unicodedata.normalize('NFC', word)
    return [vietnamese_order_dict[c] for c in word]


def vietnamese_case_insensitive_sort_key(word):
    word = unicodedata.normalize('NFC', word)
    return [vietnamese_order_dict[c] for c in word.lower()]

# pattern = "áàảãạắằẳẵặấầẩẫậèéẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
# replacement = "a"*5 + "ă"*5 + "â"*5 + "e"*5 + "ê"*5 + "i"*5 + "o"*5 + "ô"*5 + "ơ"*5 + "u"*5 + "ư"*5 + "y"*5

pattern = "ạặậ" + "ẹệ" + "ị" + "ọộợ" + "ụự" + "ỵ" #+ "ếềèé"
replacement = "aăâ" + "eê" + "i" + "oôơ" + "uư" + "y" #+ "ê"*4

BANG_XOA_DAU = str.maketrans(
    pattern,
    replacement,
)

def delete_diacritic(txt: str) -> str:
    if not unicodedata.is_normalized("NFC", txt):
        txt = unicodedata.normalize("NFC", txt)
    return txt.translate(BANG_XOA_DAU)