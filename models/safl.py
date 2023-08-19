import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.text import CharErrorRate

from .resnet_aster import ResNet_ASTER
from .decoder_transformer import AttentionRecognitionHead
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead
from .loss.sequenceCrossEntropyLoss import SequenceCrossEntropyLoss
from .utils import rule


class SAFL(pl.LightningModule):
    """
    SAFL: A Self-Attention Scene Text Recognizer with Focal Loss

    @inproceedings{tran2020safl,
        title={SAFL: A Self-Attention Scene Text Recognizer with Focal Loss},
        author={Tran, Bao Hieu and Le-Cong, Thanh and Nguyen, Huu Manh and Le, Duc Anh and Nguyen, Thanh Hung and Le Nguyen, Phi},
        booktitle={2020 19th IEEE International Conference on Machine Learning and Applications (ICMLA)},
        pages={1440--1445},
        year={2020},
        organization={IEEE}
    }
    """

    def __init__(self, num_class: int, eos: int = 187, s_dim: int = 512, att_dim: int = 512, 
                 max_len: int = 20, stn_on: bool = False,  encoder_block=4, decoder_block=4, args=None):
        """
        Arguments:
        ----------

        num_class: int
            Number of classes in the dataset

        eos: int
            End of sequence token

        s_dim: int
            Dimension of the encoder output

        att_dim: int
            Dimension of the attention layer

        max_len: int
            Maximum length of the sequence

        stn_on: bool
            Whether to use the spatial transformer network

        encoder_block: int
            Number of blocks in the encoder

        decoder_block: int
            Number of blocks in the decoder

        args:
            Arguments for training
        """
        super(SAFL, self).__init__()
        self.num_class = num_class
        self.max_len = max_len
        self.eos = eos
        self.stn_on = stn_on
        self.encoder_block = encoder_block
        self.decoder_block = decoder_block

        self.tps_inputsize = [32, 64]  # ???
        self.tps_outputsize = [32, 100]

        self.encoder = ResNet_ASTER()
        encoder_out_planes = self.encoder.out_planes

        self.decoder = AttentionRecognitionHead(
            num_classes=num_class,
            in_planes=encoder_out_planes,
            sDim=s_dim,
            attDim=att_dim,
            max_len_labels=max_len,
            encoder_block= encoder_block,
            decoder_block= decoder_block
        )

        self.criterion = SequenceCrossEntropyLoss()

        if self.stn_on:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(self.tps_outputsize),
                num_control_points=20,
                margins=tuple([0.05, 0.05]))

            self.stn_head = STNHead(
                in_planes=3,
                num_ctrlpoints=20,
                activation='none'
            )

        self.args = args
        self.example_input_array = torch.Tensor(64, 3, args.height, args.width)
        self.cer = CharErrorRate()
        self.automatic_optimization = False

    def forward(self, x):
        if self.stn_on:
            # input images are downsampled before being fed into stn_head.
            stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            stn_img_feat, ctrl_points = self.stn_head(stn_input)
            x, _ = self.tps(x, ctrl_points)

        encoder_feats = self.encoder(x)
        encoder_feats = encoder_feats.contiguous()
        preds, pred_scores = self.decoder.sample(encoder_feats)
        return preds, pred_scores
    

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        # training step
        x, targets, target_lengths = batch
        target_lengths = torch.flatten(target_lengths)

        # Compute loss
        self.train()
        if self.stn_on:
            # input images are downsampled before being fed into stn_head.
            stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            stn_img_feat, ctrl_points = self.stn_head(stn_input)
            x, _ = self.tps(x, ctrl_points)
        
        encoder_feats = self.encoder(x)
        encoder_feats = encoder_feats.contiguous()

        preds = self.decoder([encoder_feats, targets, target_lengths])
        loss = self.criterion(preds, targets, target_lengths)
        self.log('train_loss', loss, reduce_fx='mean', prog_bar=True)
        
        # Update weights
        self.manual_backward(loss)
        # clip gradients
        self.clip_gradients(opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
        opt.step()

        # Update learning rate
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
        # Log learning rate
        self.log('lr', opt.param_groups[0]['lr'], prog_bar=True)
    

    def validation_step(self, batch, batch_idx):
        x, targets, target_lengths = batch
        batch_size = x.size(0)
        target_lengths = torch.flatten(target_lengths)

        # Compute loss
        self.eval()

        if self.stn_on:
            # input images are downsampled before being fed into stn_head.
            stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            stn_img_feat, ctrl_points = self.stn_head(stn_input)
            x, _ = self.tps(x, ctrl_points)
        
        encoder_feats = self.encoder(x)
        encoder_feats = encoder_feats.contiguous()

        preds_ = self.decoder([encoder_feats, targets, target_lengths])
        preds, pred_scores = self.decoder.sample(encoder_feats)

        loss = self.criterion(preds_, targets, target_lengths)
        self.log('val_loss', loss, reduce_fx='mean', prog_bar=True)

        # Compute CER
        reals = targets.cpu().numpy().tolist()
        target_lengths = target_lengths.cpu().numpy().tolist()
        total_cer = 0
        for i, (pred, target_length) in enumerate(zip(preds, target_lengths)):
            assert target_length > 1, "Target length must be greater than 1"
            real = reals[i][ :target_length - 1]
            # Use prediction until <eos> token
            char_pred = []
            for i in pred:
                if i == self.eos:
                    break
                elif i == self.eos + 1: # <pad> token
                    continue
                char_pred.append(i)
            if len(char_pred) == 0:
                char_pred = [self.eos]
            total_cer += self.cer(torch.LongTensor(char_pred).view(1, -1), torch.LongTensor(real).view(1, -1))
        self.log('val_cer', total_cer / batch_size, reduce_fx='mean', prog_bar=True)
        print(f'val_loss: {loss}. val_cer: {total_cer / batch_size}', end='\r')


    def configure_optimizers(self):
        optimizer_params = {
            'lr': self.args.lr,
            'weight_decay': self.args.weight_decay
        }

        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), betas=(self.args.momentum, 0.98), eps=1e-9, **optimizer_params, 
            )
        elif self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), momentum=self.args.momentum, **optimizer_params
            )
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), betas=(self.args.momentum, 0.999), **optimizer_params
            )

        # Linear decay
        if self.args.warmup_steps > 0:
            # Linear scheduler
            # scheduler = optim.lr_scheduler.LambdaLR(
            #     optimizer, rule(self.args)
            # )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[4 * 39, 5 * 39], 
                gamma=0.1
            )
            return [optimizer, ], [scheduler, ]

        return optimizer