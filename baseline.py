import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.text import CharErrorRate
from torch.nn import init

from models.transformation import TPS_SpatialTransformerNetwork
from models.feature_extraction import ResNet_FeatureExtractor, VGG_FeatureExtractor, DenseNet_FeatureExtractor
from models.sequence_modeling import BidirectionalLSTM
from models.prediction import Attention
from models.decoder_transformer import TransformerRecognitionHead
from models.vitstr import create_vitstr
from utils import Averager


class Model(nn.Module):
    """
    Baseline model for CTC and Attention
    """

    def __init__(self, img_channel, img_height, img_width, num_class,
                 stn_on=False, feature_extractor='resnet', 
                 prediction='ctc', max_len=25, dropout=0.0,
                 transformer=0, transformer_model='vitstr_tiny_patch16_224'
                 ):
        """
        Arguments:
        ----------
        img_channel: int
            Number of channels of input image

        img_height: int
            Height of input image

        img_width: int
            Width of input image

        num_class: int
            Number of classes

        stn_on: bool
            Whether to use STN, default is False

        feature_extractor: str
            Feature extractor to use, either 'resnet' or 'vgg', default is 'resnet'

        prediction: str
            Prediction method to use, either 'ctc' or 'attention', default is 'ctc'

        max_len: int
            Maximum length of the sequence, default is 25

        dropout: float
            Dropout value

        transformer: int
            Whether to use ViTSTR, default is 0 (no ViTSTR)

        transformer_model: str
            ViTSTR model to use, default is 'vitstr_tiny_patch16_224'
        """
        super(Model, self).__init__()
        self.predict_method = prediction
        self.max_len = max_len

        if stn_on:
            self.tps = TPS_SpatialTransformerNetwork(
                20, (img_height, img_width), (img_height, img_width), img_channel
            )
        else:
            self.tps = nn.Identity()

        self.transformer = transformer
        if transformer:
            self.vitstr = create_vitstr(num_class, model=transformer_model)
            return

        if feature_extractor == 'resnet':
            self.feature_extractor = ResNet_FeatureExtractor(img_channel, 512)
        elif feature_extractor == 'vgg':
            self.feature_extractor = VGG_FeatureExtractor(img_channel, 512)
        elif feature_extractor == 'densenet':
            self.feature_extractor = DenseNet_FeatureExtractor(img_channel)
        
        if feature_extractor == 'densenet':
            output_channel = 2208
        else:
            output_channel = 512

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))
        if prediction == 'transformer':
            self.sequence_modeling = TransformerRecognitionHead(num_class, output_channel, max_len_labels=max_len)
        else:
            self.sequence_modeling = nn.Sequential(
                BidirectionalLSTM(output_channel, 256, 256),
                BidirectionalLSTM(256, 256, 256)
            )

        if prediction == 'ctc':
            self.prediction = nn.Linear(256, num_class)
        elif prediction == 'attention':
            self.prediction = Attention(256, 256, num_class)
        else:
            self.prediction = nn.Identity()

        # weight initialization
        initialize_weights(self.feature_extractor)
        initialize_weights(self.sequence_modeling)
        initialize_weights(self.adaptive_pool)
        initialize_weights(self.prediction)


    def forward(self, images, text=None, is_train=True, seqlen=None, tgt_padding_mask=None):
        # shape of images: (B, C, H, W)
        # Transformation
        if seqlen is None:
            seqlen = self.max_len

        images = self.tps(images)
        if self.transformer:
            prediction = self.vitstr(images, seqlen=seqlen)
            return prediction

        # Feature extraction
        visual_feature = self.feature_extractor(images)
        visual_feature = self.dropout(visual_feature)  # Dropout
        visual_feature = self.adaptive_pool(visual_feature.permute(0, 3, 1, 2)) # (B, C, H, W) -> (B, W, C, H) -> (B, W, C, 1)
        visual_feature = visual_feature.squeeze(3) # (B, W, C, 1) -> (B, W, C)

        # Sequence modeling
        if self.predict_method == 'transformer':
            contextual_feature = self.sequence_modeling(visual_feature, text, is_train, tgt_padding_mask)
        else:
            contextual_feature = self.sequence_modeling(visual_feature)

        # Prediction
        if self.predict_method == 'ctc' or self.predict_method == 'transformer':
            prediction = self.prediction(contextual_feature.contiguous())
        elif self.predict_method == 'attention':
            prediction = self.prediction(contextual_feature.contiguous(), text, is_train, seqlen)

        return prediction
    

class LightningModel(pl.LightningModule):
    """
    Lightning wrapper for baseline model
    """
    def __init__(self, model, converter, args=None):
        """
        Arguments:
        ----------
        model: nn.Module
            Model to be wrapped

        converter:
            Converter to encode and decode labels

        args: argparse.Namespace
            Arguments
        """
        super().__init__()
        self.model = model
        self.converter = converter
        self.args = args
        self.cer = CharErrorRate()
        
        if args.focal_loss:
            reduction = 'none'
        else:
            reduction = 'mean'

        if args.transformer or args.prediction != 'ctc':
            self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing, reduction=reduction)
        else:
            self.criterion = nn.CTCLoss(zero_infinity=True, reduction=reduction)
           

        # self.loss_train_avg = Averager()
        self.loss_val_avg = Averager()
        self.cer_val_avg = Averager()

        if args.train:
            dset_size = 93100
        else:
            dset_size = 103000
        if args.num_samples > 0:
            dset_size = args.num_samples

        # if args.scheduler:
        #     assert args.epochs >= args.decay_epochs, 'Number of epochs must be greater than number of decay epochs'
        num_iters = [dset_size // args.batch_size * epoch for epoch in args.decay_epochs]
        self.num_iter = num_iters
        self.automatic_optimization = False
        # self.epoch_num = 0

        if args.transformer:
            # filter that only require gradient decent
            filtered_parameters = []
            for p in filter(lambda p: p.requires_grad, self.model.parameters()):
                filtered_parameters.append(p)
            self.filtered_parameters = filtered_parameters

        if args.save:
            # Save hyperparameters
            self.save_hyperparameters()

    
    def training_step(self, batch, batch_idx):
        # Get the optimizer
        opt = self.optimizers()
        opt.zero_grad()

        # Prepare the data
        images, labels = batch
        batch_size = images.size(0)
        if not (self.args.transformer or self.args.prediction == 'transformer'):
            text, length = self.converter.encode(labels, batch_max_length=self.args.max_len)

        # Compute loss
        self.model.train()
        if self.args.transformer:
            target = self.converter.encode(labels, batch_max_length=self.args.max_len)
            preds = self.model(images, text=target, seqlen=self.converter.batch_max_length)
            loss = self.criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        elif self.args.prediction == 'ctc':
            preds = self.model(images, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.log_softmax(2).permute(1, 0, 2) # (B, T, C) -> (T, B, C)
            loss = self.criterion(preds, text, preds_size, length)
        elif self.args.prediction == 'attention':
            preds = self.model(images, text[:, :-1], True) # Align with Attention.forward
            targets = text[:, 1:] # without [GO] Symbol
            loss = self.criterion(preds.view(-1, preds.shape[-1]), targets.contiguous().view(-1))
        elif self.args.prediction == 'transformer':
            target = self.converter.encode(labels, batch_max_length=self.args.max_len)
            padding_mask = self.converter.get_padding_mask(target).to(images.device)
            preds = self.model(images, text=target[:, :-1], is_train=True, tgt_padding_mask=padding_mask)
            target = target[:, 1:] # without [GO] Symbol
            loss = self.criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        if self.args.focal_loss:
            p = torch.exp(-loss)
            loss = (self.args.focal_loss_alpha * ((1 - p) ** self.args.focal_loss_gamma) * loss).mean()

        self.log('train_loss', loss, reduce_fx='mean', prog_bar=True)
        # self.loss_train_avg.add(loss.item())

        # Update weights
        self.manual_backward(loss)
        # clip gradients
        self.clip_gradients(opt, gradient_clip_val=5, gradient_clip_algorithm="norm")
        opt.step()

        # Update learning rate
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
        # Log learning rate
        self.log('lr', opt.param_groups[0]['lr'], prog_bar=True)


    # def on_train_epoch_end(self):
    #     # print(f'Training Loss: {self.loss_train_avg.val():.4f}')
    #     # self.loss_train_avg.reset()
    #     self.epoch_num += 1


    def validation_step(self, batch, batch_idx):
        # Prepare the data
        images, labels = batch
        batch_size = images.size(0)
        if self.args.transformer or self.args.prediction == 'transformer':
            target = self.converter.encode(labels, batch_max_length=self.args.max_len)
        else:
            length_for_pred = torch.IntTensor([self.args.max_len] * batch_size)
            text_for_pred = torch.LongTensor(batch_size, self.args.max_len + 1).fill_(0)
            text_for_loss, length_for_loss = self.converter.encode(labels, batch_max_length=self.args.max_len)

        # Compute loss
        self.model.eval()
        if self.args.transformer:
            preds = self.model(images, text=target, seqlen=self.converter.batch_max_length)
            _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
            preds_index = preds_index.view(-1, self.converter.batch_max_length)

            val_loss = self.criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            length_for_pred = torch.IntTensor([self.converter.batch_max_length - 1] * batch_size)
            preds_str = self.converter.decode(preds_index[:, 1:], length_for_pred)
        elif self.args.prediction == 'ctc':
            preds = self.model(images, text_for_pred)
            pred_size = torch.IntTensor([preds.size(1)] * batch_size)
            val_loss = self.criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, pred_size, length_for_loss)
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index.data, pred_size.data)
        elif self.args.prediction == 'attention':
            preds = self.model(images, text_for_pred, False)
            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:] # without [GO] Symbol
            val_loss = self.criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            labels = self.converter.decode(text_for_loss[:, 1:], length_for_loss)
        elif self.args.prediction == 'transformer':
            preds = self.model(images, is_train=False)
            _, preds_index = preds.max(2)
            val_loss = self.criterion(preds.view(-1, preds.shape[-1]), target[:, 1:-1].contiguous().view(-1))
            length_for_pred = torch.IntTensor([self.converter.batch_max_length - 1] * batch_size)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        if self.args.focal_loss:
            p = torch.exp(-val_loss)
            val_loss = (self.args.focal_loss_alpha * ((1 - p) ** self.args.focal_loss_gamma) * val_loss).mean()

        self.log('val_loss', val_loss, reduce_fx='mean', prog_bar=True)
        self.loss_val_avg.add(val_loss.item())

        # Compute CER
        total_cer = 0
        for gt, pred in zip(labels, preds_str):
            if self.args.transformer or self.args.prediction == 'transformer':
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]
            elif self.args.prediction == 'attention':
                pred_EOS = pred.find('[s]')
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred_EOS]

            total_cer += self.cer([pred], [gt])
        val_cer = total_cer / batch_size
        self.log('val_cer', val_cer, reduce_fx='mean', prog_bar=True)
        self.cer_val_avg.add(val_cer.item())


    def on_validation_epoch_end(self):
        print(f'Validation Loss: {self.loss_val_avg.val():.4f}. Validation CER: {self.cer_val_avg.val():.4f}')
        self.loss_val_avg.reset()
        self.cer_val_avg.reset()


    def configure_optimizers(self):
        optimizer_params = {
            'lr': self.args.lr,
            'weight_decay': self.args.weight_decay
        }

        if self.args.transformer:
            params = self.filtered_parameters
        else:
            params = self.parameters()

        if self.args.optim == 'adam':
            optimizer = torch.optim.Adam(
                params, betas=(self.args.momentum, 0.999), **optimizer_params
            )
        elif self.args.optim == 'sgd':
            optimizer = torch.optim.SGD(
                params, momentum=self.args.momentum, **optimizer_params
            )
        elif self.args.optim == 'adamw':
            optimizer = torch.optim.AdamW(
                params, betas=(self.args.momentum, 0.999), **optimizer_params
            )
        elif self.args.optim == 'adadelta':
            optimizer = torch.optim.Adadelta(
                params, **optimizer_params, eps=1e-8, rho=0.95
            )

        # Learning rate scheduler
        if self.args.scheduler:
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer, T_max=self.num_iter
            # )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.num_iter, gamma=self.args.decay_rate
            )
            return [optimizer, ], [scheduler, ]

        return optimizer
    


def initialize_weights(model):
    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue