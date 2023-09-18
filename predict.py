import numpy as np
import pandas as pd

from config import LABEL_FILE
from utils import delete_diacritic


def make_test_prediction(candidates, mask=False, alpha=0.125, case_sensitive=True, diacritics=False):
    frame = []
    for candidate in candidates:
        frame.append(pd.read_csv(f'predictions/{candidate}.csv', na_filter=False))

    # Vote based on confidence
    confidences = np.array([frame[i]['confidence'] for i in range(len(frame))]).T
    predictions = np.array([frame[i]['pred'] for i in range(len(frame))]).T
    if mask:
        labels = pd.read_csv(LABEL_FILE, header=None, na_filter=False, encoding='utf-8', sep='\t')
        labels.columns = ['id', 'label']
        if case_sensitive:
            vocab = labels['label'].unique()
            conf_mask = np.isin(predictions, vocab).astype(int)
        else:
            vocab = pd.Series(labels['label'].str.lower().unique())
            conf_mask = np.zeros_like(predictions)
            if diacritics:
                vocab = pd.Series([delete_diacritic(vocab[i]) for i in range(len(vocab))]).unique()
                for i in range(predictions.shape[0]):
                    for j in range(predictions.shape[1]):
                        conf_mask[i, j] = delete_diacritic(predictions[i, j].lower()) in vocab
            else:
                for i in range(predictions.shape[1]):
                    conf_mask[:, i] = np.isin(pd.Series(predictions[:, i]).str.lower(), vocab).astype(int)
        confidences = confidences + alpha * conf_mask

    idx = np.argmax(confidences, axis=1)
    pred = pd.Series([predictions[i, idx[i]] for i in range(len(idx))])

    # Save prediction
    df = pd.DataFrame({'img_name': frame[0]['img_name'], 'pred': pred})
    df.to_csv('predictions/prediction.txt', index=False, header=False, sep='\t')

    return pred

if __name__ == '__main__':
    print('-' * 20)
    print('Inference')
    winning_candidates = [
        'model5_synth_full', 'model9_full', 'model7_full',
        'model4_full', 'model15_full', 'model10_synth_full',
        'model10_full', 'model3_full',
        'model4_synth_full', 'model2_synth_full', 'model5_full',
        ]

    pred = make_test_prediction(winning_candidates, mask=True, alpha=0.5625, case_sensitive=False)

    print('Done!')
    print('-' * 20)