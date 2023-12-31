import torch
import torch.nn.functional as F
import pickle
import pandas as pd

from baseline import Model
from models.parseq import PARSeq
from models.abinet import ABINet
from models.satrn import SATRN
from models.corner import CornerTransformer
from torchmetrics.text import CharErrorRate
from argparse import ArgumentParser
from .converters import CTCLabelConverter, AttnLabelConverter, TokenLabelConverter, SRNConverter, ParseqConverter, CPPDConverter, WordArtConverter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_arguments(argv):
    parser = ArgumentParser()
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed, default: 42')
    parser.add_argument('--val_check_interval', type=float, default=0.5, help='Validation check interval per epoch, default: 0.5')
    parser.add_argument('--model_name', type=str, help='Name of the model to train')
    parser.add_argument('--save', type=int, default=0, help='Whether to save for training later or not, default: 0 (not save)')
    parser.add_argument('--pin_memory', type=int, default=1, help='Whether to use pin memory or not, default: 1 (use pin memory)')

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
    parser.add_argument('--focal_loss_alpha', type=float, default=1.0, help='Alpha for focal loss, default: 0.5')

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
    parser.add_argument('--synth_augment', type=int, default=0, help='Whether to use SynthText augmentation or not, default: 0 (not use SynthText augmentation)')

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

    # ABINet
    parser.add_argument('--abinet_v_backbone', type=str, default='transformer', help='V-backbone for ABINet, default: transformer, options: transformer, resnet')

    # SVTR
    parser.add_argument('--svtr_backbone', type=str, default='large', help='Backbone for SVTR, default: large, options: large, base')

    # Other models
    parser.add_argument('--feature_extractor', type=str, default='resnet', help='Feature extractor, default: resnet, options: resnet, vgg, densenet, aster, convnext, svtr, vit-base, vit-small')
    parser.add_argument('--stn_on', type=int, default=0, help='Whether to use STN or not, default: 0 (not use STN)')
    parser.add_argument('--prediction', type=str, default='ctc', help='Prediction method, default: ctc, options: ctc, attention, srn, parseq, abinet, cppd, satrn, corner')
    parser.add_argument('--max_len', type=int, default=25, help='Max length of the predicted text, default: 25')
    parser.add_argument('--seq_dim', type=int, default=256, help='Dimension of the sequence encoder, default: 256')

    # Auxiliary loss
    parser.add_argument('--count_mark', type=int, default=0, help='Whether to count mark or not, default: 0 (not count mark)')
    parser.add_argument('--mark_alpha', type=float, default=1.0, help='Alpha for mark, default: 1.0')
    parser.add_argument('--count_case', type=int, default=0, help='Whether to count uppercase or not, default: 0 (not count mark)')
    parser.add_argument('--case_alpha', type=float, default=1.0, help='Alpha for uppercase, default: 1.0')
    parser.add_argument('--count_char', type=int, default=0, help='Whether to count number of characters or not, default: 0 (not count number of characters)')
    parser.add_argument('--char_alpha', type=float, default=1.0, help='Alpha for number of characters, default: 1.0')

    # Labeling
    parser.add_argument('--tone', type=int, default=0, help='Whether to use tone or not, default: 0 (not use tone)')

    return parser.parse_args(argv)
    


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


def load_model(name, model_path='saved_models'):
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
    device = get_device()

    # Get the converter
    converter = build_converter(args)
    model = build_model(args, converter)
    model.load_state_dict(torch.load(f'{model_path}/{name}.pt'))
    model.eval()
    model.to(device)

    return model, converter, args


def predict_train_valid(model, converter, data_loader, args):
    
    cer = CharErrorRate()
    device = get_device()
    preds_lst = []
    reals_lst = []
    cer_lst = []
    confidences = []
    max_length = args.max_len
    transformer = args.transformer
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            # Prepare data
            images, labels, _, _ = batch
            images = images.to(device)
            batch_size = images.size(0)

            # Predict
            preds, preds_str = predict_batch(model, converter, images, batch_size, transformer, max_length, args.prediction)

            # Compute confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            # Compute CER
            for i, (gt, pred, pred_max_prob) in enumerate(zip(labels, preds_str, preds_max_prob)):
                pred, confidence_score = postprocess(pred, pred_max_prob, transformer, args.prediction)
                confidences.append(confidence_score)
                preds_lst.append(pred)
                reals_lst.append(gt)
                cer_lst.append(cer([pred], [gt]).item())

    return preds_lst, reals_lst, cer_lst, confidences


def build_converter(args):
    # Get the converter
    if args.transformer:
        converter = TokenLabelConverter(args.max_len, args.tone)
    elif args.prediction == 'ctc':
        converter = CTCLabelConverter(args.tone)
    elif args.prediction == 'attention':
        converter = AttnLabelConverter(args.tone)
    elif args.prediction == 'srn':
        converter = SRNConverter(args.tone)
    elif args.prediction == 'parseq' or args.prediction == 'abinet':
        converter = ParseqConverter(args.tone)
    elif args.prediction == 'cppd':
        converter = CPPDConverter(args.tone)
    elif args.prediction in ['satrn', 'corner']:
        converter = WordArtConverter(args.tone)

    return converter


def build_model(args, converter):
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
    elif args.prediction == 'abinet':
        model = ABINet(
            args.max_len, NUM_CLASSES, converter.pad_id, converter.bos_id, converter.eos_id, 
            args.weight_decay, v_backbone=args.abinet_v_backbone,
        )
    elif args.prediction == 'satrn':
        model = SATRN(
            input_channel, converter, NUM_CLASSES, args.max_len + 2
        )
    elif args.prediction == 'corner':
        model = CornerTransformer(
            input_channel, converter, NUM_CLASSES, args.max_len + 2
        )
    else:
        model = Model(
            input_channel, args.height, args.width, NUM_CLASSES,
            args.stn_on, args.feature_extractor, args.prediction,
            dropout=args.dropout, max_len=args.max_len, 
            transformer=args.transformer, transformer_model=args.transformer_model,
            svtr_backbone=args.svtr_backbone, seq_dim=args.seq_dim,
        )

    return model


def predict_batch(model, converter, images, batch_size, transformer, max_length, prediction):
    if not transformer:
        length_for_pred = torch.IntTensor([max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, max_length + 1).fill_(0).to(device)

    # Compute loss
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
    elif prediction == 'parseq' or prediction == 'abinet':
        preds = model(images)
        _, preds_index = preds.max(2) # (B, T, C) -> (B, T), greedy decoding
        preds_str = converter.decode(preds_index, length_for_pred)
    elif prediction == 'cppd':
        preds, _ = model(images)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
    elif prediction == 'satrn':
        preds, _ = model(images, train_mode=False)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
    elif prediction == 'corner':
        preds, _ = model(images, train_mode=False)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

    return preds, preds_str


def postprocess(pred, pred_max_prob, transformer, prediction):
    if transformer or prediction == 'attention':
        pred_EOS = pred.find('[s]')
        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
        pred_max_prob = pred_max_prob[:pred_EOS]
    elif prediction in ['srn', 'parseq', 'abinet', 'cppd', 'satrn', 'corner']:
        pred_EOS = len(pred)
        pred_max_prob = pred_max_prob[:pred_EOS]
    try:
        confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
    except:
        confidence_score = 0.0  # Case when pred_max_prob is empty

    return pred, confidence_score

