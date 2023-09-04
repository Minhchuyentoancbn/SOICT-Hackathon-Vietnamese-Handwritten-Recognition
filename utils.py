import torch
import pickle
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_denmark(text):
    """
    Count the number for each type of denmark in a string.

    Arguments:
    ----------
    text: list(str)
        List of strings.

    Returns:
    --------
    marks: torch.Tensor(N, 5)
        The number of each type of denmark in the string.
    """
    sac = list('ÁÉÍÓÚÝáéíóúýẤấẮắẾếỐốỚớỨứ')
    huy = list('ÀÈÌÒÙỲàèìòùỳẦầẰằỀềỒồỜờỪừ')
    nga = list('ÃẼĨÕŨỸãẽĩõũỹẪẫẴẵỄễỖỗỠỡỮữ')
    nan = list('ẠẸỊỌỤỴạẹịọụỵẬậẶặỆệỘộỢợỰự')
    hoi = list('ẢẺỈỎỦỶảẻỉỏủỷẨẩẲẳỂểỔổỞởỬử')

    N = len(text)
    marks = torch.zeros(N, 5)
    for i, t in enumerate(text):
        for c in t:
            if c in sac:
                marks[i, 0] += 1
            elif c in huy:
                marks[i, 1] += 1
            elif c in nga:
                marks[i, 2] += 1
            elif c in nan:
                marks[i, 3] += 1
            elif c in hoi:
                marks[i, 4] += 1
    return marks


def count_uppercase(text):
    """
    Count the number of uppercase characters in a string.

    Arguments:
    ----------
    text: list(str)
        List of strings.

    Returns:
    --------
    uppercase: torch.Tensor(N, 1)
        The number of uppercase characters in the string.
    """
    N = len(text)
    uppercase = torch.zeros(N, 1)
    for i, t in enumerate(text):
        for c in t:
            if c.isupper():
                uppercase[i, 0] += 1
    return uppercase


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