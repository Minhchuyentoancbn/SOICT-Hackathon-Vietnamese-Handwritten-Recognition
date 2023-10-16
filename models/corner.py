import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
import math

from models.mmcv.backbone import ShallowCNN
from models.mmcv.layers import CornerCrossattnLayer, Adaptive2DPositionalEncoding, PositionalEncoding, TFDecoderLayer
from models.satrn import TFLoss


class CornerPreprocessor(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxCorners = 200
        self.qualityLevel = 0.01
        self.minDistance = 3

    def forward(self, batch_img):
        """
        Args:
            batch_img (Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Corner map with size :math:`(N, 1, H, W)`.
        """
        device = batch_img.device
        img_np = batch_img.cpu().numpy()
        batch_corner_map = torch.Tensor()
        for i in range(img_np.shape[0]):
            sin_img = img_np[i].transpose(1, 2 ,0) * 255
            gray_img = cv2.cvtColor(sin_img, cv2.COLOR_BGR2GRAY)
            gray_img = np.float32(gray_img)
            img_bg = np.zeros(gray_img.shape, dtype="uint8")
            corners = cv2.goodFeaturesToTrack(gray_img, self.maxCorners, self.qualityLevel, self.minDistance)
            try:
                corners = np.int0(corners)
                for corner in corners:
                    x,y = corner.ravel()
                    img_bg[y,x] = 1
            except TypeError:
                print('No corner detected!')
            
            corner_mask = torch.tensor(img_bg).unsqueeze(0).unsqueeze(0)
            corner_mask = corner_mask.to(torch.float32)
            batch_corner_map = torch.cat([batch_corner_map, corner_mask], dim=0)
        batch_corner_map = batch_corner_map.to(device)
        return batch_corner_map
    

class CornerEncoder(nn.Module):
    """Implement encoder for SATRN, see `SATRN.

    <https://arxiv.org/abs/1910.04396>`_.

    Args:
        n_layers (int): Number of attention layers.
        n_head (int): Number of parallel attention heads.
        d_k (int): Dimension of the key vector.
        d_v (int): Dimension of the value vector.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
        n_position (int): Length of the positional encoding vector. Must be
            greater than ``max_seq_len``.
        d_inner (int): Hidden dimension of feedforward layers.
        dropout (float): Dropout rate.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 n_layers=12,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 n_position=100,
                 d_inner=512 * 4,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        self.d_model = d_model
        self.position_enc = Adaptive2DPositionalEncoding(
            d_hid=d_model,
            n_height=n_position,
            n_width=n_position,
            dropout=dropout)
        self.position_enc_corner = Adaptive2DPositionalEncoding(
            d_hid=d_model,
            n_height=n_position,
            n_width=n_position,
            dropout=dropout)
    
        self.layer_stack = nn.ModuleList([
            CornerCrossattnLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, feat, corner_feat):
        """
        Args:
            feat (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.

        Returns:
            Tensor: A tensor of shape :math:`(N, T, D_m)`.
        """
        valid_ratios = [1.0 for _ in range(feat.size(0))]
        feat = self.position_enc(feat)
        corner_feat = self.position_enc_corner(corner_feat)
        
        n, c, h, w = feat.size()
        mask = feat.new_zeros((n, h, w))
        for i, valid_ratio in enumerate(valid_ratios):
            valid_width = min(w, math.ceil(w * valid_ratio))
            mask[i, :, :valid_width] = 1
        mask = mask.view(n, h * w)
        feat = feat.view(n, c, h * w)
        corner_feat = corner_feat.view(n, c, h * w)

        output = feat.permute(0, 2, 1).contiguous()
        corner = corner_feat.permute(0, 2, 1).contiguous()
        for enc_layer in self.layer_stack:
            output = enc_layer(output, corner, h, w, mask)
        output = self.layer_norm(output)

        return output
    

class SupConHead(nn.Module):
    """backbone + projection head"""

    def __init__(self):
        super(SupConHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        )

    def forward(self, x):
        bs, length = x.shape[:2]
        x = x.reshape(-1, 512)
        x = self.head(x)
        feat = F.normalize(x, dim=1)

        feat = feat.reshape(bs, length, 512)
        return feat
    

class CharContDecoder(nn.Module):
    """Transformer Decoder block with self attention mechanism.

    Args:
        n_layers (int): Number of attention layers.
        d_embedding (int): Language embedding dimension.
        n_head (int): Number of parallel attention heads.
        d_k (int): Dimension of the key vector.
        d_v (int): Dimension of the value vector.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
        d_inner (int): Hidden dimension of feedforward layers.
        n_position (int): Length of the positional encoding vector. Must be
            greater than ``max_seq_len``.
        dropout (float): Dropout rate.
        num_classes (int): Number of output classes :math:`C`.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        start_idx (int): The index of `<SOS>`.
        padding_idx (int): The index of `<PAD>`.
        init_cfg (dict or list[dict], optional): Initialization configs.

    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss as specified in
        :obj:`mmocr.models.textrecog.recognizer.EncodeDecodeRecognizer`.
    """

    def __init__(self,
                 n_layers=6,
                 d_embedding=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=512 * 4,
                 n_position=200,
                 dropout=0.1,
                 num_classes=93,
                 max_seq_len=40,
                 start_idx=1,
                 padding_idx=92,
                 **kwargs):
        super().__init__()

        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.max_seq_len = max_seq_len

        self.trg_word_emb = nn.Embedding(
            num_classes, d_embedding, padding_idx=padding_idx)

        self.position_enc = PositionalEncoding(
            d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            TFDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout, **kwargs)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        pred_num_class = num_classes - 1  # ignore padding_idx
        self.classifier = nn.Linear(d_model, pred_num_class)
        self.cc_head = SupConHead()

    @staticmethod
    def get_pad_mask(seq, pad_idx):

        return (seq != pad_idx).unsqueeze(-2)

    @staticmethod
    def get_subsequent_mask(seq):
        """For masking out the subsequent info."""
        len_s = seq.size(1)
        subsequent_mask = 1 - torch.triu(
            torch.ones((len_s, len_s), device=seq.device), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).bool()

        return subsequent_mask

    def _attention(self, trg_seq, src, src_mask=None):
        trg_embedding = self.trg_word_emb(trg_seq)
        trg_pos_encoded = self.position_enc(trg_embedding)
        tgt = self.dropout(trg_pos_encoded)

        trg_mask = self.get_pad_mask(
            trg_seq,
            pad_idx=self.padding_idx) & self.get_subsequent_mask(trg_seq)
        output = tgt
        for dec_layer in self.layer_stack:
            output = dec_layer(
                output,
                src,
                self_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
        output = self.layer_norm(output)

        return output

    def _get_mask(self, logit):
        N, T, _ = logit.size()
        valid_ratios = [1.0 for _ in range(N)]
        mask = None
        if valid_ratios is not None:
            mask = logit.new_zeros((N, T))
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(T, math.ceil(T * valid_ratio))
                mask[i, :valid_width] = 1

        return mask

    def forward_train(self, feat, out_enc, targets):
        r"""
        Args:
            feat (None): Unused.
            out_enc (Tensor): Encoder output of shape :math:`(N, T, D_m)`
                where :math:`D_m` is ``d_model``.
            targets: a tensor of shape :math:`(N, T)`. Each element is the index of a character.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, T, C)`.
        """
        src_mask = self._get_mask(out_enc)
        targets = targets.to(out_enc.device)
        attn_output = self._attention(targets, out_enc, src_mask=src_mask)
        outputs = self.classifier(attn_output)
        char_outputs = self.cc_head(attn_output)

        return outputs, char_outputs

    def forward_test(self, feat, out_enc):
        src_mask = self._get_mask(out_enc)
        N = out_enc.size(0)
        init_target_seq = torch.full((N, self.max_seq_len + 1),
                                     self.padding_idx,
                                     device=out_enc.device,
                                     dtype=torch.long)
        # bsz * seq_len
        init_target_seq[:, 0] = self.start_idx

        outputs = []
        for step in range(0, self.max_seq_len):
            decoder_output = self._attention(
                init_target_seq, out_enc, src_mask=src_mask)
            # bsz * seq_len * C
            step_result = self.classifier(decoder_output[:, step, :])
            # bsz * num_classes
            outputs.append(step_result)
            _, step_max_index = torch.max(step_result, dim=-1)
            init_target_seq[:, step + 1] = step_max_index

        outputs = torch.stack(outputs, dim=1)

        return outputs, None

    def forward(self,
                feat,
                out_enc,
                targets=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets)

        return self.forward_test(feat, out_enc)




class CornerTransformer(nn.Module):
    def __init__(self, img_channels, converter, num_classes, max_len=25):
        super().__init__()
        self.converter = converter

        self.preprocessor = CornerPreprocessor()
        self.backbone = ShallowCNN(input_channels=img_channels, hidden_dim=512)
        self.backbone_corner = ShallowCNN(input_channels=1, hidden_dim=512)
        self.encoder = CornerEncoder()
        self.decoder = CharContDecoder(num_classes=num_classes, max_seq_len=max_len, 
                                       padding_idx=converter.pad_id, start_idx=converter.bos_id)


    def extract_feat(self, img):
        """Directly extract features from the backbone."""
        mask = self.preprocessor(img)
        x = self.backbone(img)
        x_corner = self.backbone_corner(mask)

        return x, x_corner
    
    def forward(self, img, targets=None, train_mode=False):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
        """

        feat, corner = self.extract_feat(img)

        out_enc = self.encoder(feat, corner)

        out_dec, char_out_dec = self.decoder(
            feat, out_enc, targets, train_mode=train_mode)

        return out_dec, char_out_dec
    

class CornerLoss(nn.Module):
    def __init__(self, ignore_index=-1, smoothing=0, reduction='mean'):
        super().__init__()
        self.loss = TFLoss(ignore_index=ignore_index, smoothing=smoothing, reduction=reduction)
        self.cc_loss = CCLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, out_dec, char_out_dec, targets):
        loss = self.loss(out_dec, targets)
        if char_out_dec is not None:
            char_loss = self.cc_loss(char_out_dec, targets)
            loss = loss + char_loss

        return loss



class CCLoss(nn.Module):
    """Implementation of loss module for encoder-decoder based text recognition
    method with Character Contrastive loss.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
        ignore_first_char (bool): Whether to ignore the first token in target (
            usually the start token). If ``True``, the last token of the output
            sequence will also be removed to be aligned with the target length.
    """

    def __init__(self,
                 ignore_index=-1,
                 reduction='mean',
                 ignore_first_char=False):
        super().__init__()
        assert isinstance(ignore_index, int)
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        assert isinstance(ignore_first_char, bool)
        self.loss = SupConLoss()
        self.flatten = True

    def format(self, outputs, targets):
        outputs = outputs[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()
        if self.flatten:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()

        return outputs, targets

    def forward(self, outputs, targets):
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            targets: A dict with a key ``padded_targets``, which is
                a tensor of shape :math:`(N, T)`. Each element is the index of
                a character.

        Returns:
            dict: A loss dict with the key ``loss_cc``.
        """
        outputs, targets = self.format(outputs, targets)
        outputs = outputs.unsqueeze(dim=1)

        loss = self.loss(outputs, targets.to(outputs.device))
        # loss_sup = self.sup_contra_loss(outputs, targets.to(outputs.device))
       
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.1, contrast_mode='one',
                 base_temperature=0.07, ignore_index=-1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.ignore_index = ignore_index


    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. 

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        ori_labels = labels

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:

            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)

            mask[:, ori_labels == self.ignore_index] = 0
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        logits_mask[:, ori_labels == self.ignore_index] = 0
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        loss = 0.1 * loss

        return loss
