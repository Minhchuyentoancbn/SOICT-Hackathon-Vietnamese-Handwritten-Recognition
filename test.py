import torch
import torch.nn.functional as F

from utils import get_device


def predict(model, dataloader, coverter, prediction, max_length=25):
    """
    Predict the result of the model

    Arguments:
    ----------
    model: torch.nn.Module
        The model used to predict

    dataloader: torch.utils.data.DataLoader
        The dataloader of the dataset

    coverter: AttnLabelConverter
        The converter used to convert the label to character

    prediction: str
        The prediction method

    max_length: int
        The maximum length of the sequence

    Returns:
    --------
    all_preds: list
        The list of the prediction result

    img_names_lst: list
        The list of the image names

    confidences: list
        The list of the confidence of the prediction
    """
    
    all_preds = []
    img_names_lst = []
    confidences = []

    device = get_device()
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for data, img_names in dataloader:
            # Prepare data
            batch_size = data.size(0)
            images = data.to(device)
            length_for_pred = torch.IntTensor([max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, max_length + 1).fill_(0).to(device)

            # Predict
            if prediction == 'ctc':
                preds = model(images, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2) # (B, T, C) -> (B, T), greedy decoding
                preds_str = coverter.decode(preds_index, preds_size)
            else:
                preds = model(images, text_for_pred, is_train=False)
                _, preds_index = preds.max(2) # (B, T, C) -> (B, T), greedy decoding
                preds_str = coverter.decode(preds_index, length_for_pred)

            all_preds += preds_str
            img_names_lst += list(img_names)

            # Compute confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                if prediction == 'attention':
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]
                    
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                confidences.append(confidence_score.item())

    return all_preds, img_names_lst, confidences