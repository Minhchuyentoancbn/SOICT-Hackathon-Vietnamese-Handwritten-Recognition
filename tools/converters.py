import torch
import random
from .text_tools import tone_decode, tone_encode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, tone=False):
        self.to_tone = tone
        # character (str): set of the possible characters.
        character = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        if tone:
            character = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
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
        if self.to_tone:
            text = [tone_encode(t) for t in text]
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
            if self.to_tone:
                text = tone_decode(text)
            texts.append(text)
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, tone=False):
        self.to_tone = tone
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        character = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        if tone:
            character = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
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
        if self.to_tone:
            text = [tone_encode(t) for t in text]
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
            if self.to_tone:
                text = tone_decode(text)
            texts.append(text)
        return texts
    

class TokenLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, max_len=25, tone=False):
        self.to_tone = tone
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        character = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        if tone:
            character = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
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
        if self.to_tone:
            text = [tone_encode(t) for t in text]
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
            if self.to_tone:
                text = tone_decode(text)
            texts.append(text)
        return texts
    

class SRNConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, tone=False):
        self.to_tone = tone
        # character (str): set of the possible characters.
        character = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ$#'
        if tone:
            character = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$#'
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
        if self.to_tone:
            text = [tone_encode(t) for t in text]
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
            text = text[:idx]
            if self.to_tone:
                text = tone_decode(text)
            texts.append(text)
        return texts
    

class ParseqConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, tone=False):
        max_len = 25
        self.to_tone = tone
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        character = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        if tone:
            character = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
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
        if self.to_tone:
            text = [tone_encode(t) for t in text]
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
            text = text[:idx]
            if self.to_tone:
                text = tone_decode(text)
            texts.append(text)
        return texts
    
    def get_tgt_paddding_mask(self, text, batch_max_length=25):
        """ 
        Create target padding mask for Transformer.
        """
        batch_text = self.encode(text, batch_max_length)
        tgt_mask = batch_text == self.pad_id
        tgt_mask = tgt_mask.to(device)
        # tgt_mask = tgt_mask[:, :-1]
        return batch_text, tgt_mask
    

class CPPDConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, tone=False, ignore_index=200):
        max_len = 25
        self.to_tone = tone
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        character = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        if tone:
            character = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self.EOS = '[E]'
        self.BOS = '[B]'

        self.character = ['</s>'] + list(character)
        self.num_classes = len(self.character)

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = max_len
        self.ignore_index = ignore_index


    def encode(self, text, batch_max_length=25):
        """ 
        Convert text-label into text-index.
        """
        if self.to_tone:
            text = [tone_encode(t) for t in text]

        length = [len(s) for s in text]
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(self.ignore_index)
        label_node = []

        for i, t in enumerate(text):
            txt, txt_node = self.text_encode(t)
            txt_pos_node = [1] * (len(txt) + 1) + [0] * (self.batch_max_length - len(txt))
            txt.append(0)  # eos
            txt = txt + [self.ignore_index] * (self.batch_max_length + 1 - len(txt))
            label_node.append(txt_node + txt_pos_node)
            # print(len(txt_node + txt_pos_node))
            batch_text[i][:len(txt)] = torch.LongTensor(txt)
        return batch_text.to(device), torch.IntTensor(label_node).to(device), torch.IntTensor(length).to(device)
    

    def text_encode(self, text):
        text_list = []
        text_node = [0 for _ in range(self.num_classes)]
        text_node[0] = 1

        for char in text:
            i_c = self.dict[char]
            text_list.append(i_c)
            text_node[i_c] += 1

        return text_list, text_node


    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            idx = text.find('</s>')
            text = text[:idx]
            if self.to_tone:
                text = tone_decode(text)
            texts.append(text)
        return texts
