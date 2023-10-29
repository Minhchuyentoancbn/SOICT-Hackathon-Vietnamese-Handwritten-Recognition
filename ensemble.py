import numpy as np
import pandas as pd
import os

from torchmetrics.text import CharErrorRate
from config import LABEL_FILE, PUBLIC_TEST_DIR
from collections import Counter
from tools import delete_diacritic
from dataset import count_denmark, count_uppercase
from PIL import Image


def get_model_frame(total_models: list, set_name='val', idx=-1):
    # Read the data
    model_frame = dict()
    for model in total_models:
        if idx < 0:
            model_frame[model] = pd.read_csv(f'ensemble/{set_name}/{model}.csv', na_filter=False)
        else:
            model_frame[model] = pd.read_csv(f'ensemble/{set_name}/{model}.csv', na_filter=False).iloc[:idx]
    return model_frame


def compute_vote_cer(frame, models: list, mask=None):
    # Vote based on confidence
    confidences = np.array([frame[model]['confidence'] for model in models]).T
    cer = np.array([frame[model]['cer'] for model in models]).T
    if mask is not None:
        confidences = confidences + mask
    predictions = np.array([frame[model]['pred'] for model in models]).T
    idx = np.argmax(confidences, axis=1)
    pred = predictions[np.arange(len(idx)), idx]
    # real = frame[models[0]]['real']
    cer_score = np.mean(cer[np.arange(len(idx)), idx])
    
    return cer_score, pred


def compute_vote_char_cer(frame, models, mode='hard'):
    preds = np.array([frame[model]['pred'] for model in models]).T
    conf = np.array([frame[model]['confidence'] for model in models]).T
    preds_vote = []
    for pred in preds:
        # Find most common length
        if mode == 'hard':
            c = Counter([len(p) for p in pred])
            most_common_len = c.most_common(1)[0][0]
            pred_vote = ''
            for i in range(most_common_len):
                c = Counter([p[i] for p in pred if len(p) > i])
                pred_vote += c.most_common(1)[0][0]
        elif mode == 'soft':
            c = Counter([len(p) for p in pred])
            most_common_len = c.most_common(1)[0][0]
            pred_vote = ''
            for i in range(most_common_len):
                char_dict = dict()
                for j in range(len(models)):
                    if len(pred[j]) <= i:
                        continue
                    char = pred[j][i]
                    if char in char_dict:
                        char_dict[char] += conf[i, j]
                    else:
                        char_dict[char] = conf[i, j]
                pred_vote += max(char_dict, key=char_dict.get)
        preds_vote.append(pred_vote)

    cer_func = CharErrorRate()
    reals = frame[models[0]]['real']
    cer_scores = []
    for i in range(len(reals)):
        cer_scores.append(cer_func([preds_vote[i]], [reals[i]]))
    cer = np.mean(cer_scores)
    return cer, np.array(preds_vote)



def make_final_prediction(word_candidates, char_based_pred_full, alpha=1.0, beta=0.25, set_name='test'):
    labels = pd.read_csv(LABEL_FILE, header=None, na_filter=False, encoding='utf-8', sep='\t')
    labels.columns = ['id', 'label']
    # vocab = labels['label'].unique()
    vocab = labels['label'].str.lower().unique()
    vocab = pd.Series([delete_diacritic(label) for label in vocab]).unique()
    vocab_dict = {word: 1 for word in vocab}
    test_model_frame = get_model_frame(word_candidates, set_name=set_name)
    preds = np.array([test_model_frame[model]['pred'] for model in word_candidates]).T
    confidences = np.array([test_model_frame[model]['confidence'] for model in word_candidates]).T
    test_mask = np.zeros_like(preds)
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            test_mask[i, j] = vocab_dict.get(delete_diacritic(preds[i, j].lower()), 0) * alpha
            if preds[i, j] == char_based_pred_full[i]:
                test_mask[i, j] += beta

    scores = confidences + test_mask

    idx = np.argmax(scores, axis=1)
    pred = pd.Series([preds[i, idx[i]] for i in range(len(idx))])

    # Save prediction
    df = pd.DataFrame({'img_name': test_model_frame[word_candidates[0]]['img_name'], 'pred': pred})
    df.to_csv('ensemble/prediction.txt', index=False, header=False, sep='\t')

    win_conf = confidences[np.arange(len(idx)), idx]
    df = pd.DataFrame({'img_name': test_model_frame[word_candidates[0]]['img_name'], 'pred': pred, 'confidence': win_conf})
    df.to_csv('ensemble/prediction.csv', index=False)

    return pred


def make_final_char_prediction(char_candidates, mode='soft', set_name='test'):
    test_model_frame = get_model_frame(char_candidates, set_name=set_name)
    preds = np.array([test_model_frame[model]['pred'] for model in char_candidates]).T
    conf= np.array([test_model_frame[model]['confidence'] for model in char_candidates]).T
    preds_vote = []
    for pred in preds:
        # Find most common length
        if mode == 'hard':
            c = Counter([len(p) for p in pred])
            most_common_len = c.most_common(1)[0][0]
            pred_vote = ''
            for i in range(most_common_len):
                c = Counter([p[i] for p in pred if len(p) > i])
                pred_vote += c.most_common(1)[0][0]
        elif mode == 'soft':
            c = Counter([len(p) for p in pred])
            most_common_len = c.most_common(1)[0][0]
            pred_vote = ''
            for i in range(most_common_len):
                char_dict = dict()
                for j in range(len(char_candidates)):
                    if len(pred[j]) <= i:
                        continue
                    char = pred[j][i]
                    if char in char_dict:
                        char_dict[char] += conf[i, j]
                    else:
                        char_dict[char] = conf[i, j]
                pred_vote += max(char_dict, key=char_dict.get)
        preds_vote.append(pred_vote)

    # Save prediction
    df = pd.DataFrame({'img_name': test_model_frame[char_candidates[0]]['img_name'], 'pred': preds_vote})
    df.to_csv('ensemble/char_prediction.txt', index=False, header=False, sep='\t')

    return np.array(preds_vote)
    

def add_full_to_lst(models):
    models_full = []
    for model in models:
        if os.path.exists(f'saved_models/{model}_full.pt'):
            models_full.append(model + '_full')
        else:
            models_full.append(model)
    return models_full