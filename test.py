import torch
import torch.nn.functional as F

from utils import get_device


def predict(model, dataloader, converter, prediction, max_length=25, transformer=0):
    """
    Predict the result of the model

    Arguments:
    ----------
    model: torch.nn.Module
        The model used to predict

    dataloader: torch.utils.data.DataLoader
        The dataloader of the dataset

    converter: AttnLabelConverter
        The converter used to convert the label to character

    prediction: str
        The prediction method

    max_length: int
        The maximum length of the sequence

    transformer: int
        Whether to use transformer

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
            if not transformer:
                length_for_pred = torch.IntTensor([max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, max_length + 1).fill_(0).to(device)

            # Predict
            if transformer:
                preds = model(images, seqlen=converter.batch_max_length, text=None)
                _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
                preds_index = preds_index.view(-1, converter.batch_max_length)
                length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
                preds_str = converter.decode(preds_index[:, 1:], length_for_pred)
            elif prediction == 'ctc':
                preds, _ = model(images, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2) # (B, T, C) -> (B, T), greedy decoding
                preds_str = converter.decode(preds_index, preds_size)
            elif prediction == 'attention':
                preds, _ = model(images, text_for_pred, is_train=False)
                _, preds_index = preds.max(2) # (B, T, C) -> (B, T), greedy decoding
                preds_str = converter.decode(preds_index, length_for_pred)
            elif prediction == 'srn':
                preds, _ = model(images, None)
                preds = preds[2]
                _, preds_index = preds.max(2) # (B, T, C) -> (B, T), greedy decoding
                preds_str = converter.decode(preds_index, length_for_pred)
            elif prediction == 'parseq':
                preds = model(images)
                _, preds_index = preds.max(2) # (B, T, C) -> (B, T), greedy decoding
                preds_str = converter.decode(preds_index, length_for_pred)

            img_names_lst += list(img_names)

            # Compute confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                if transformer or prediction == 'attention':   
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]
                elif prediction == 'srn' or prediction == 'parseq':
                    pred_EOS = len(pred)
                    pred_max_prob = pred_max_prob[:pred_EOS]
                    
                all_preds.append(pred)
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    confidences.append(confidence_score.item())
                except:
                    confidence_score = 0.0  # Case when pred_max_prob is empty
                    confidences.append(confidence_score)

    return all_preds, img_names_lst, confidences