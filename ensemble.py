# Implementation of Weight-Adjusted Voting algorithm for Ensembles of Classifiers (WAVE)

import numpy as np
import pandas as pd
import torch

from torchmetrics.text import CharErrorRate
from itertools import combinations


def compute_weights(model_frames, models_lst):
    """
    Computes the weights for each model in the ensemble.

    Arguments:
    ----------
    model_frames: dict(str, pd.DataFrame)
        Dictionary of the dataframes for each model.

    models_lst: list[str]
        List of the names of the models.

    Returns:
    --------
    weights: np.array
        Array of the weights for each model.
    """

    reals = model_frames[models_lst[0]]['real']
    predictions = np.array([model_frames[model]['pred'] for model in models_lst])
    n = len(reals)
    k = len(models_lst)
    X = np.zeros((n, k))

    for i in range(k):
        X[:, i] = (reals == predictions[i]).astype(int)
    
    # Define two matrices consisting of 1s
    J_nk = np.ones((n, k))
    J_kk = np.ones((k, k))

    # Define an identity matrix
    identity_k = np.identity(k)
    
    # Compute the matrix T 
    T = X.T.dot((J_nk-X)).dot((J_kk - identity_k))
    
    # Find eigenvalues and eigenvectors of T 
    eig_values, eig_vectors = np.linalg.eig(T)[0].real, np.linalg.eig(T)[1].real

    # Find r domain eigenvalues 
    max_eig_value = eig_values.max()
    r = 0
    idxes_max_eig = []
    for i in range(len(eig_values)):
        if eig_values[i] == max_eig_value:
            r += 1
            idxes_max_eig.append(i)
    
    # compute matrix sigma
    sigma = np.zeros((k, k))
    for i in range(r):
        u = eig_vectors[:, idxes_max_eig[i]]
        u = u.reshape((k, 1))
        sigma += u.dot(u.T)
    
    # Define a vector of 1s
    k_1 = np.ones((k, 1))
    
    # Compute the weight vector and set it as self.weights
    weights = (sigma.dot(k_1)) / k_1.T.dot(sigma).dot(k_1)

    return weights.squeeze()


def make_test_prediction(candidates):
    frame = []
    for candidate in candidates:
        frame.append(pd.read_csv(f'ensemble/test/{candidate}.csv', na_filter=False))

    # Vote based on confidence
    confidences = np.array([frame[i]['confidence'] for i in range(len(frame))]).T
    predictions = np.array([frame[i]['pred'] for i in range(len(frame))]).T
    idx = np.argmax(confidences, axis=1)
    pred = pd.Series([predictions[i, idx[i]] for i in range(len(idx))])

    # Save prediction
    df = pd.DataFrame({'img_name': frame[0]['img_name'], 'pred': pred})
    df.to_csv('ensemble/prediction.txt', index=False, header=False, sep='\t')

    return pred


def compute_vote_cer_aug(frames):
    # Vote based on confidence
    confidences = np.array([frame['confidence'] for frame in frames]).T
    predictions = np.array([frame['pred'] for frame in frames]).T
    idx = np.argmax(confidences, axis=1)
    pred = [predictions[i, idx[i]] for i in range(len(idx))]
    real = frames[0]['real']
    
    # Compute CER
    cer = CharErrorRate()
    cer_score = cer(pred, real)

    return cer_score


def get_model_frame(total_models: list, set_name='val', idx=-1):
    # Read the data
    model_frame = dict()
    for model in total_models:
        if idx < 0:
            model_frame[model] = pd.read_csv(f'ensemble/{set_name}/{model}.csv', na_filter=False)
        else:
            model_frame[model] = pd.read_csv(f'ensemble/{set_name}/{model}.csv', na_filter=False).iloc[:idx]
    return model_frame


def compute_vote_cer(frame, models: list, weight=None):
    # Vote based on confidence
    confidences = np.array([frame[model]['confidence'] for model in models]).T
    if weight is not None:
        confidences = np.log(confidences) * weight
    predictions = np.array([frame[model]['pred'] for model in models]).T
    idx = np.argmax(confidences, axis=1)
    pred = [predictions[i, idx[i]] for i in range(len(idx))]
    real = frame[models[0]]['real']
    
    # Compute CER
    cer = CharErrorRate()
    cer_score = cer(pred, real)

    return cer_score


def find_best_comb(num_voters, total_models, model_frame):
    # Iterate through all subsets
    best_cer = 1.0
    best_subset = None

    for subset in combinations(total_models, num_voters):
        cer = compute_vote_cer(model_frame, list(subset))
        if cer < best_cer:
            best_cer = cer
            best_subset = subset

    print(f'Best subset: {best_subset}')
    print(f'CER Train: {best_cer}')

    return best_subset, best_cer