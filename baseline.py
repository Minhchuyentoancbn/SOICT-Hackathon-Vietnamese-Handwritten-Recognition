import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import timm
from torchmetrics.text import CharErrorRate
from torch.nn import init

from models.transformation import TPS_SpatialTransformerNetwork
from models.feature_extraction import ResNet_FeatureExtractor, VGG_FeatureExtractor, DenseNet_FeatureExtractor
from models.sequence_modeling import BidirectionalLSTM
from models.prediction import Attention
from models.srn import Transforme_Encoder, SRN_Decoder, cal_performance
from models.counter import MarkCounter, UpperCaseCounter
from models.vitstr import create_vitstr
from models.resnet_aster import ResNet_ASTER
from models.svtr import SVTRNet
from models.cppd import CPPDHead, CPPDLoss
from timm.optim import create_optimizer_v2



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
        self.extractor_type = feature_extractor

        output_stn_size = (img_height, img_width)

        # Transformation
        if stn_on:
            input_size = (img_height, img_width)
            if feature_extractor == 'svtr' and prediction != 'cppd':
                output_stn_size = (32, 100)
            self.tps = TPS_SpatialTransformerNetwork(
                20, input_size, output_stn_size, img_channel
            )
        else:
            self.tps = nn.Identity()

        # For ViTSTR
        self.transformer = transformer
        if transformer:
            self.vitstr = create_vitstr(num_class, model=transformer_model)
            return

        # Feature extraction
        if feature_extractor == 'svtr':
            if prediction == 'cppd':
                last_stage = False
                prenorm = True
            else:
                last_stage = True
                prenorm = False
            self.feature_extractor = SVTRNet(
                img_size=output_stn_size,
                in_channels=img_channel,
                embed_dim=[192, 256, 512],
                depth=[3, 9, 9],
                num_heads=[6, 8, 16],
                out_channels=384,
                mixer=['Local'] * 10 + ['Global'] * 11,
                last_stage=last_stage, prenorm=prenorm,
            )
        elif feature_extractor == 'resnet':
            self.feature_extractor = ResNet_FeatureExtractor(img_channel, 512)
        elif feature_extractor == 'vgg':
            self.feature_extractor = VGG_FeatureExtractor(img_channel, 512)
        elif feature_extractor == 'densenet':
            self.feature_extractor = DenseNet_FeatureExtractor(img_channel)
        elif feature_extractor == 'aster':
            self.feature_extractor = ResNet_ASTER(img_channel)
        elif feature_extractor == 'convnext':
            self.feature_extractor = timm.create_model(
                'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320', 
                pretrained=True, features_only=True,
            )
    
        # Output channel of the feature extractor
        if feature_extractor == 'densenet':
            output_visual_channel = 2208
        elif feature_extractor == 'convnext':
            output_visual_channel = 768
        elif feature_extractor == 'svtr' and prediction != 'cppd':
            output_visual_channel = 384
        else:
            output_visual_channel = 512

        # Dropout and adaptive pooling after feature extraction
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        if feature_extractor != 'svtr':
            self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))
        else:
            self.adaptive_pool = nn.Identity()

        # Sequence modeling
        if prediction == 'srn':
            self.sequence_modeling = Transforme_Encoder(output_visual_channel, n_position=img_width // 4 + 1)
        elif feature_extractor == 'svtr':
            self.sequence_modeling = nn.Identity()
        else:
            self.sequence_modeling = nn.Sequential(
                BidirectionalLSTM(output_visual_channel, 256, 256),
                BidirectionalLSTM(256, 256, 256)
            )

        # Output dimension of the sequence modeling
        if prediction == 'srn':
            output_seq_dim = img_width // 4 + 1
        elif feature_extractor == 'svtr':
            output_seq_dim = output_visual_channel
        else:
            output_seq_dim = 256

        # Prediction
        if prediction == 'ctc':
            self.prediction = nn.Linear(output_seq_dim, num_class)
        elif prediction == 'attention':
            self.prediction = Attention(output_seq_dim, output_seq_dim, num_class)
        elif prediction == 'srn':
            self.prediction = SRN_Decoder(n_position=img_width // 4 + 1, N_max_character=max_len + 1, n_class=num_class)
        elif prediction == 'cppd':
            if feature_extractor == 'svtr':
                vis_seq = img_width * img_height // 64
            elif feature_extractor == 'resnet':
                vis_seq = img_width // 4 + 1
            self.prediction = CPPDHead(output_seq_dim, num_class, max_len=max_len, dim=output_seq_dim, vis_seq=vis_seq)
        else:
            self.prediction = nn.Identity()

        # weight initialization
        if feature_extractor != 'convnext' and feature_extractor != 'svtr':
            initialize_weights(self.feature_extractor)
        initialize_weights(self.sequence_modeling)
        initialize_weights(self.adaptive_pool)
        initialize_weights(self.prediction)


    def forward(self, images, text=None, is_train=True, seqlen=None):
        # shape of images: (B, C, H, W)
        # Transformation
        if seqlen is None:
            seqlen = self.max_len

        images = self.tps(images)

        # For ViTSTR
        if self.transformer:
            prediction = self.vitstr(images, seqlen=seqlen)
            return prediction

        # Feature extraction
        if self.extractor_type == 'convnext':
            feature_map = self.feature_extractor(images)[-2]
        else:
            feature_map = self.feature_extractor(images)
        visual_feature = self.dropout(feature_map)  # Dropout
        if self.extractor_type != 'svtr':
            visual_feature = self.adaptive_pool(visual_feature.permute(0, 3, 1, 2)) # (B, C, H, W) -> (B, W, C, H) -> (B, W, C, 1)
            visual_feature = visual_feature.squeeze(3) # (B, W, C, 1) -> (B, W, C)
        elif self.predict_method == 'cppd':
            visual_feature = visual_feature.permute([0, 2, 1])

        # Sequence modeling
        if self.predict_method == 'srn':
            contextual_feature = self.sequence_modeling(visual_feature, src_mask=None)[0]
        else:
            contextual_feature = self.sequence_modeling(visual_feature)

        # Prediction
        if self.predict_method == 'ctc':
            prediction = self.prediction(contextual_feature.contiguous())
        elif self.predict_method == 'attention':
            prediction = self.prediction(contextual_feature.contiguous(), text, is_train, seqlen)
        elif self.predict_method == 'srn' or self.predict_method == 'cppd':
            prediction = self.prediction(contextual_feature)

        return prediction, feature_map
    

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

        # Additonal modules for counting diacritics, uppercase letters, and characters
        if (not args.transformer):
            if args.feature_extractor == 'densenet':
                output_channel = 2208
            else:
                output_channel = 512
            if args.prediction == 'parseq':
                output_channel = 384

            if args.count_mark:
                self.mark_counter = MarkCounter(output_channel)
                self.mark_crit = nn.MSELoss()
                initialize_weights(self.mark_counter)
            if args.count_case or args.count_char:
                self.case_counter = UpperCaseCounter(output_channel)
                self.case_crit = nn.MSELoss()
                self.char_crit = nn.MSELoss()
                initialize_weights(self.case_counter)
        
        # For focal loss
        if args.focal_loss:
            reduction = 'none'
        else:
            reduction = 'mean'

        # Criterion
        if args.transformer or args.prediction == 'attention':
            self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing, reduction=reduction)
        elif args.prediction == 'ctc':
            self.criterion = nn.CTCLoss(zero_infinity=True, reduction=reduction)
        elif args.prediction == 'srn':
            self.criterion = cal_performance
            if args.label_smoothing > 0:
                self.smoothing = '0'
            else:
                self.smoothing = '1'
        elif args.prediction == 'parseq' or args.prediction == 'abinet':
            self.criterion = nn.CrossEntropyLoss(ignore_index=converter.pad_id, reduction=reduction, label_smoothing=args.label_smoothing)
        elif args.prediction == 'cppd':
            self.criterion = CPPDLoss(args.label_smoothing > 0, converter.ignore_index)

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

        # For ViTSTR, filter out parameters that do not require gradient decent
        if args.transformer:
            # filter that only require gradient decent
            filtered_parameters = []
            for p in filter(lambda p: p.requires_grad, self.model.parameters()):
                filtered_parameters.append(p)
            self.filtered_parameters = filtered_parameters

        # Save hyperparameters
        if args.save: 
            self.save_hyperparameters()

    
    def training_step(self, batch, batch_idx):
        # Get the optimizer
        opt = self.optimizers()
        opt.zero_grad()

        # Prepare the data
        images, labels, num_marks, num_uppercase = batch
        batch_size = images.size(0)
        if not (self.args.transformer or self.args.prediction in ['parseq', 'abinet', 'cppd']):
            text, length = self.converter.encode(labels, batch_max_length=self.args.max_len)

        # Compute loss
        self.model.train()
        if self.args.transformer:
            target = self.converter.encode(labels, batch_max_length=self.args.max_len)
            preds = self.model(images, text=target, seqlen=self.converter.batch_max_length)
            loss = self.criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        elif self.args.prediction == 'ctc':
            preds, visual_feature = self.model(images, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.log_softmax(2).permute(1, 0, 2) # (B, T, C) -> (T, B, C)
            loss = self.criterion(preds, text, preds_size, length)
        elif self.args.prediction == 'attention':
            preds, visual_feature = self.model(images, text[:, :-1], True) # Align with Attention.forward
            targets = text[:, 1:] # without [GO] Symbol
            loss = self.criterion(preds.view(-1, preds.shape[-1]), targets.contiguous().view(-1))
        elif self.args.prediction == 'srn':
            preds, visual_feature = self.model(images, None)
            loss = self.criterion(preds, text, self.converter.PAD, self.smoothing)
        elif self.args.prediction == 'parseq':
            target = self.converter.encode(labels, batch_max_length=self.args.max_len)
            
            # Encode the source sequence (i.e. the image codes)
            memory = self.model.encode(images)

            # Prepare the target sequences (input and output)
            tgt_perms = self.model.gen_tgt_perms(target)
            tgt_in = target[:, :-1]
            tgt_out = target[:, 1:]
            # The [EOS] token is not depended upon by any other token in any permutation ordering
            tgt_padding_mask = (tgt_in == self.model.pad_id) | (tgt_in == self.model.eos_id)

            loss = 0
            loss_numel = 0
            n = (tgt_out != self.model.pad_id).sum().item()
            for i, perm in enumerate(tgt_perms):
                tgt_mask, query_mask = self.model.generate_attn_masks(perm)
                out = self.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
                preds = self.model.head(out).flatten(end_dim=1)
                loss += n * self.criterion(preds, tgt_out.flatten())
                loss_numel += n
                # After the second iteration (i.e. done with canonical and reverse orderings),
                # remove the [EOS] tokens for the succeeding perms
                if i == 1:
                    tgt_out = torch.where(tgt_out == self.model.eos_id, self.model.pad_id, tgt_out)
                    n = (tgt_out != self.model.pad_id).sum().item()
            loss /= loss_numel
        elif self.args.prediction == 'abinet':
            inputs, lengths, targets = self.model._prepare_inputs_and_targets(labels, self.converter)
            if self._pretraining:
                v_res = self.model.model.vision(images)
                l_res = self.model.model.language(inputs, lengths)
                a_res = self.model.model.alignment(l_res['feature'].detach(), v_res['feature'].detach())
                loss = self.model.calc_loss(self.criterion, targets, v_res, l_res, a_res)
            else:
                if self.model._reset_aligment:
                    self.model.reset_alignment()
                all_a_res, all_l_res, v_res = self.model.model.forward(images)
                loss = self.model.calc_loss(self.criterion, targets, v_res, all_l_res, all_a_res)
        elif self.args.prediction == 'cppd':
            target_batch = self.converter.encode(labels, batch_max_length=self.args.max_len)
            preds, _ = self.model(images)
            loss = self.criterion(preds, target_batch)['loss']

        # Focal loss
        if self.args.focal_loss:
            p = torch.exp(-loss)
            loss = (self.args.focal_loss_alpha * ((1 - p) ** self.args.focal_loss_gamma) * loss).mean()

        # Count diacritics, uppercase letters, and characters
        if (not self.args.transformer):
            if (self.args.count_mark or self.args.count_case or self.args.count_char) and self.args.prediction == 'parseq' and (not self.args.parseq_use_transformer):
                visual_feature = self.model.get_visual_features(images)
            if self.args.count_mark:
                mark_pred = self.mark_counter(visual_feature)
                mark_loss = self.mark_crit(mark_pred, num_marks)
                loss = mark_loss * self.args.mark_alpha + loss
            if self.args.count_case or self.args.count_char:
                case_pred, num_char_pred = self.case_counter(visual_feature)
                if self.args.count_case:
                    case_loss = self.case_crit(case_pred, num_uppercase)
                else:
                    case_loss = 0
                if self.args.count_char:
                    char_loss = self.char_crit(num_char_pred, length.float())
                else:
                    char_loss = 0
                loss = loss + case_loss * self.args.case_alpha  + char_loss * self.args.char_alpha

        # Log training loss
        self.log('train_loss', loss, reduce_fx='mean', prog_bar=True)
        # self.loss_train_avg.add(loss.item())

        # Update weights
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=self.args.clip_grad_val, gradient_clip_algorithm="norm")
        opt.step()

        # Update learning rate
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
        # Log learning rate
        self.log('lr', opt.param_groups[0]['lr'], prog_bar=True)


    def validation_step(self, batch, batch_idx):
        # Prepare the data
        images, labels, num_marks, num_uppercase = batch
        batch_size = images.size(0)
        if self.args.transformer or self.args.prediction in ['parseq', 'abinet']:
            target = self.converter.encode(labels, batch_max_length=self.args.max_len)
        elif self.args.prediction != 'cppd':
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
            preds, visual_feature = self.model(images, text_for_pred)
            pred_size = torch.IntTensor([preds.size(1)] * batch_size)
            val_loss = self.criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, pred_size, length_for_loss)
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index.data, pred_size.data)
        elif self.args.prediction == 'attention':
            preds, visual_feature = self.model(images, text_for_pred, False)
            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:] # without [GO] Symbol
            val_loss = self.criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            # labels = self.converter.decode(text_for_loss[:, 1:], length_for_loss)
        elif self.args.prediction == 'srn':
            preds, visual_feature = self.model(images, None)
            val_loss = self.criterion(preds, text_for_loss, self.converter.PAD, self.smoothing)
            _, preds_index = preds[2].max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            # labels = self.converter.decode(text_for_loss, length_for_loss)
        elif self.args.prediction == 'parseq':
            length_for_pred = torch.IntTensor([self.args.max_len] * batch_size)
            target = target[:, 1:]
            max_len = target.shape[1] - 1
            preds = self.model.forward(images, max_len)
            val_loss = self.criterion(preds.flatten(end_dim=1), target.flatten())
            # loss_numel = (target != self.model.pad_id).sum().item()
            # val_loss /= loss_numel
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
        elif self.args.prediction == 'abinet':
            length_for_pred = torch.IntTensor([self.args.max_len] * batch_size)
            target = target[:, 1:]
            max_len = target.shape[1] - 1
            preds = self.model.forward(images, max_len)
            val_loss = self.criterion(preds.flatten(end_dim=1), target.flatten())
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
        elif self.args.prediction == 'cppd':
            length_for_pred = torch.IntTensor([self.args.max_len] * batch_size)
            target_batch = self.converter.encode(labels, batch_max_length=self.args.max_len)
            self.model.train()
            with torch.no_grad():
                preds_train, _ = self.model(images)
            val_loss = self.criterion(preds_train, target_batch)['loss']
            self.model.eval()
            preds, _ = self.model(images)
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        # Focal loss
        if self.args.focal_loss:
            p = torch.exp(-val_loss)
            val_loss = (self.args.focal_loss_alpha * ((1 - p) ** self.args.focal_loss_gamma) * val_loss).mean()

        # Count diacritics, uppercase letters, and characters
        if (not self.args.transformer) and self.args.prediction != 'parseq':
            if self.args.count_mark:
                mark_pred = self.mark_counter(visual_feature)
                mark_loss = self.mark_crit(mark_pred, num_marks)
                val_loss = mark_loss * self.args.mark_alpha + val_loss
            if self.args.count_case or self.args.count_char:
                case_pred, num_char_pred = self.case_counter(visual_feature)
                if self.args.count_case:
                    case_loss = self.case_crit(case_pred, num_uppercase)
                else:
                    case_loss = 0
                if self.args.count_char:
                    char_loss = self.char_crit(num_char_pred, length_for_loss)
                else:
                    char_loss = 0
                val_loss = val_loss + case_loss * self.args.case_alpha  + char_loss * self.args.char_alpha

        # Log validation loss
        self.log('val_loss', val_loss, reduce_fx='mean', prog_bar=True)
        self.loss_val_avg.add(val_loss.item())

        # Compute CER
        total_cer = 0
        for gt, pred in zip(labels, preds_str):
            if self.args.transformer or self.args.prediction == 'attention':
                pred_EOS = pred.find('[s]')
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
        # Optimizer
        if not self.args.timm_optim:
            # Common hyperparameters for optimizer
            optimizer_params = {
                'lr': self.args.lr,
                'weight_decay': self.args.weight_decay
            }

            # Filter out parameters
            if self.args.transformer:
                params = self.filtered_parameters
            elif self.args.prediction == 'parseq':
                params = self.model.parameters()
            elif self.args.prediction == 'abinet':
                params = self.model.learnable_params()
            else:
                params = self.parameters()

            # Create optimizer
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
        else:  # Use timm optimizer
            optimizer = create_optimizer_v2(self.model, self.args.optim, 
                                            lr=self.args.lr, weight_decay=self.args.weight_decay,
                                            momentum=self.args.momentum
                                            )

        # Learning rate scheduler
        if self.args.scheduler:
            if self.args.one_cycle:
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.args.lr, total_steps=self.trainer.estimated_stepping_batches, 
                    pct_start=0.075, cycle_momentum=False
                )
            else:
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=self.num_iter, gamma=self.args.decay_rate
                )
            return [optimizer, ], [scheduler, ]

        return optimizer
    
    @property
    def _pretraining(self):
        # In the original work, VM was pretrained for 8 epochs while full model was trained for an additional 10 epochs.
        total_steps = self.trainer.estimated_stepping_batches
        return self.global_step < (8 / (8 + 10)) * total_steps
    

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


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        # count = v.data.numel()
        # v = v.data.sum()
        self.n_count += 1#count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
