import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.text import CharErrorRate

import sys
from .utils import initialize_weights, rule
from .transformation import TPS_SpatialTransformerNetwork
from .resnet import ResNet_FeatureExtractor
from .sequence_modeling import BidirectionalLSTM
from .prediction import Attention
from .utils import parse_arguments

global_args = parse_arguments(sys.argv[1:])

class AttentionEncoderDecoder(pl.LightningModule):
    """
    Attention Encoder-Decoder model.
    - Image feature extractor: ResNet
    - Sequence modeling: Bidirectional LSTM
    - Prediction: Attention
    """

    def __init__(self, img_height, img_width, num_class, converter, max_len=20, label_smoothing=0.0, dropout=0.0):
        """
        Arguments:
        ----------

        img_height: int
            Height of input image

        img_width: int
            Width of input image

        num_class: int
            Number of classes

        converter: AttnLabelConverter
            Converter used to convert the label to character

        max_len: int
            Maximum length of the sequence

        label_smoothing: float
            Label smoothing value

        dropout: float
            Dropout value
        """
        super(AttentionEncoderDecoder, self).__init__()

        if global_args.stn_on:
            self.tps = TPS_SpatialTransformerNetwork(
                20, (img_height, img_width), (img_height, img_width), 3
            )
        else:
            self.tps = nn.Identity()


        self.feature_extractor, (output_channel, output_height, output_width) = self.resnet_backbone(3, img_height, img_width)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))

        self.sequence_modeling = nn.Sequential(
            BidirectionalLSTM(output_channel, 256, 256),
            BidirectionalLSTM(256, 256, 256)
        )

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        initialize_weights(self.feature_extractor)
        initialize_weights(self.sequence_modeling)
        initialize_weights(self.adaptive_pool)

        self.prediction = Attention(256, 256, num_class)

        self.max_len = max_len
        self.converter = converter
        self.cer = CharErrorRate()
        self.automatic_optimization = False
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)


    def resnet_backbone(self, img_channel, img_height, img_width):
        output_channel = 512
        output_height = img_height // 16 - 1
        output_width = img_width // 4 + 1
        cnn = ResNet_FeatureExtractor(img_channel, 512)
        return cnn, (output_channel, output_height, output_width)


    def forward(self, images, text):
        # shape of images: (B, C, H, W)

        # Transformation
        images = self.tps(images)

        # Feature extraction
        visual_feature = self.feature_extractor(images)
        visual_feature = self.dropout(visual_feature)
        visual_feature = self.adaptive_pool(visual_feature.permute(0, 3, 1, 2)) # (B, C, H, W) -> (B, W, C, H) -> (B, W, C, 1)
        visual_feature = visual_feature.squeeze(3) # (B, W, C, 1) -> (B, W, C)

        # Sequence modeling
        contextual_feature = self.sequence_modeling(visual_feature)

        # Prediction
        prediction = self.prediction(contextual_feature.contiguous(), text, self.training, self.max_len)
        return prediction
    

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        # training step
        images, labels = batch
        batch_size = images.size(0)
        text, length = self.converter.encode(labels, batch_max_length=self.max_len)

        # Compute loss
        self.train()
        preds = self.forward(images, text[:, :-1]) # Align with Attention.forward
        targets = text[:, 1:] # without [GO] Symbol
        loss = self.criterion(preds.view(-1, preds.shape[-1]), targets.contiguous().view(-1))
        self.log('train_loss', loss, reduce_fx='mean', prog_bar=True)
        
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


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        batch_size = images.size(0)

        # For max length prediction
        length_for_pred = torch.IntTensor([self.max_len] * batch_size).to(self.device)
        text_for_pred = torch.LongTensor(batch_size, self.max_len + 1).fill_(0)

        text_for_loss, length_for_loss = self.converter.encode(labels, batch_max_length=self.max_len)

        self.eval()
        with torch.no_grad():
            preds = self.forward(images, text_for_pred)
        preds = preds[:, :text_for_loss.shape[1] - 1, :]
        target = text_for_loss[:, 1:]  # without [GO] Symbol
        
        val_loss = self.criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
        # greedy decode
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, length_for_pred)
        labels = self.converter.decode(text_for_loss[:, 1:], length_for_loss)

        self.log('val_loss', val_loss, reduce_fx='mean', prog_bar=True)

        total_cer = 0
        for gt, pred in zip(labels, preds_str):
            gt = gt[:gt.find('[s]')]
            pred = pred[:pred.find('[s]')]
            total_cer += self.cer([pred], [gt])

        self.log('val_cer', total_cer / batch_size, reduce_fx='mean', prog_bar=True)
        print(f'val_loss: {val_loss}. val_cer: {total_cer / batch_size}', end='\r')


    def configure_optimizers(self):
        optimizer_params = {
            'lr': global_args.lr,
            'weight_decay': global_args.weight_decay
        }

        if global_args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), betas=(global_args.momentum, 0.999), **optimizer_params
            )
        elif global_args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), momentum=global_args.momentum, **optimizer_params
            )
        elif global_args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), betas=(global_args.momentum, 0.999), **optimizer_params
            )
        elif global_args.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(
                self.parameters(), **optimizer_params, eps=1e-8, rho=0.95
            )

        # Linear warmup
        if global_args.warmup_steps > 0:
            # Linear scheduler
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, rule(global_args)
            )
            return [optimizer, ], [scheduler, ]

        return optimizer