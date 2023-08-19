import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
from utils import get_device
from models.ctc_decoder import ctc_decode


def predict_ctc(model, dataloader, label2char, decode_method, beam_size):
    """
    Predict the result of the model

    Arguments:
    ----------
    model: torch.nn.Module
        The model used to predict

    dataloader: torch.utils.data.DataLoader
        The dataloader of the dataset

    label2char: dict
        The dictionary of the label to character

    decode_method: str
        The method used to decode the result

    beam_size: int
        The beam size of the beam search

    Returns:
    --------
    all_preds: list
        The list of the prediction result
    """
    model.eval()

    all_preds = []
    img_names = []
    device = get_device()
    model = model.to(device)

    with torch.no_grad():
        for data, img_name in dataloader:
            images = data.to(device)
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            all_preds += preds
            img_names += list(img_name)

    all_preds = [''.join(pred) for pred in all_preds]
    return all_preds, img_names


import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
from utils import get_device
from models.ctc_decoder import ctc_decode


def predict_safl(model, dataloader, label2char, eos, pad):
    """
    Predict the result of the model

    Arguments:
    ----------
    model: torch.nn.Module
        The model used to predict

    dataloader: torch.utils.data.DataLoader
        The dataloader of the dataset

    label2char: dict
        The dictionary of the label to character

    eos: int
        The id of the end of sentence token

    pad: int
        The id of the padding token

    Returns:
    --------
    all_preds: list
        The list of the prediction result
    """
    model.eval()

    all_preds = []
    img_names = []
    device = get_device()
    model = model.to(device)

    with torch.no_grad():
        for data, img_name in dataloader:
            images = data.to(device)
            preds, pred_scores = model(images)
            detection_list = []
            for pred in preds:
                detection = []
                for char in pred:
                    p = char.item()
                    if p == eos:
                        break
                    elif p == pad:
                        continue
                    else:
                        detection.append(label2char[p])
                detection_list.append(detection)

            all_preds += detection_list
            img_names += list(img_name)

    all_preds = [''.join(pred) for pred in all_preds]
    return all_preds, img_names