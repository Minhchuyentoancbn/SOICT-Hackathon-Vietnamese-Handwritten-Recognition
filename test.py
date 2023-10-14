import torch
import torch.nn.functional as F

from tools import get_device, predict_batch, postprocess


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

            # Predict
            preds, preds_str = predict_batch(model, converter, images, batch_size, transformer, max_length, prediction)

            img_names_lst += list(img_names)

            # Compute confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                pred, confidence_score = postprocess(pred, pred_max_prob, transformer, prediction)
                confidences.append(confidence_score)
                all_preds.append(pred)

    return all_preds, img_names_lst, confidences