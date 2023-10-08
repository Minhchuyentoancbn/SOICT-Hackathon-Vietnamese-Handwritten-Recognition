import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Optional
from timm.optim.optim_factory import param_groups_weight_decay
from .transformer import PositionalEncoding, TransformerDecoderLayer
from .attention import PositionAttention, Attention
from .backbone import ResTranformer
from .resnet import resnet45


class BaseModel(nn.Module):

    def __init__(self, dataset_max_length: int, null_label: int):
        super().__init__()
        self.max_length = dataset_max_length + 1  # additional stop token
        self.null_label = null_label

    def _get_length(self, logit, dim=-1):
        """ Greed decoder to obtain length from logit"""
        out = (logit.argmax(dim=-1) == self.null_label)
        abn = out.any(dim)
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1  # additional end token
        out = torch.where(abn, out, out.new_tensor(logit.shape[1], device=out.device))
        return out

    @staticmethod
    def _get_padding_mask(length, max_length):
        length = length.unsqueeze(-1)
        grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        return grid >= length

    @staticmethod
    def _get_location_mask(sz, device=None):
        mask = torch.eye(sz, device=device)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask


class BaseAlignment(BaseModel):
    def __init__(self, dataset_max_length, null_label, num_classes, d_model=512, loss_weight=1.0):
        super().__init__(dataset_max_length, null_label)
        self.loss_weight = loss_weight
        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature 
        """
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'logits': logits, 'pt_lengths': pt_lengths, 'loss_weight': self.loss_weight,
                'name': 'alignment'}


class BCNLanguage(BaseModel):
    def __init__(self, dataset_max_length, null_label, num_classes, d_model=512, nhead=8, d_inner=2048, dropout=0.1,
                 activation='relu', num_layers=4, detach=True, use_self_attn=False, loss_weight=1.0,
                 global_debug=False):
        super().__init__(dataset_max_length, null_label)
        self.detach = detach
        self.loss_weight = loss_weight
        self.proj = nn.Linear(num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout,
                                                activation, self_attn=use_self_attn, debug=global_debug)
        self.model = nn.TransformerDecoder(decoder_layer, num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach:
            tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed,
                            tgt_key_padding_mask=padding_mask,
                            memory_mask=location_mask,
                            memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res = {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
               'loss_weight': self.loss_weight, 'name': 'language'}
        return res


class BaseVision(BaseModel):
    def __init__(self, dataset_max_length, null_label, num_classes,
                 attention='position', attention_mode='nearest', loss_weight=1.0,
                 d_model=512, nhead=8, d_inner=2048, dropout=0.1, activation='relu',
                 backbone='transformer', backbone_ln=2):
        super().__init__(dataset_max_length, null_label)
        self.loss_weight = loss_weight
        self.out_channels = d_model

        if backbone == 'transformer':
            self.backbone = ResTranformer(d_model, nhead, d_inner, dropout, activation, backbone_ln)
        else:
            self.backbone = resnet45()

        if attention == 'position':
            self.attention = PositionAttention(
                max_length=self.max_length,
                mode=attention_mode
            )
        elif attention == 'attention':
            self.attention = Attention(
                max_length=self.max_length,
                n_feature=8 * 32,
            )
        else:
            raise ValueError(f'invalid attention: {attention}')

        self.cls = nn.Linear(self.out_channels, num_classes)

    def forward(self, images):
        features = self.backbone(images)  # (N, E, H, W)
        attn_vecs, attn_scores = self.attention(features)  # (N, T, E), (N, T, H, W)
        logits = self.cls(attn_vecs)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'loss_weight': self.loss_weight, 'name': 'vision'}


class ABINetIterModel(nn.Module):
    def __init__(self, dataset_max_length, null_label, num_classes, iter_size=1,
                 d_model=512, nhead=8, d_inner=2048, dropout=0.1, activation='relu',
                 v_loss_weight=1., v_attention='position', v_attention_mode='nearest',
                 v_backbone='transformer', v_num_layers=2,
                 l_loss_weight=1., l_num_layers=4, l_detach=True, l_use_self_attn=False,
                 a_loss_weight=1.):
        super().__init__()
        self.iter_size = iter_size
        self.vision = BaseVision(dataset_max_length, null_label, num_classes, v_attention, v_attention_mode,
                                 v_loss_weight, d_model, nhead, d_inner, dropout, activation, v_backbone, v_num_layers)
        self.language = BCNLanguage(dataset_max_length, null_label, num_classes, d_model, nhead, d_inner, dropout,
                                    activation, l_num_layers, l_detach, l_use_self_attn, l_loss_weight)
        self.alignment = BaseAlignment(dataset_max_length, null_label, num_classes, d_model, a_loss_weight)


    def forward(self, images):
        v_res = self.vision(images)
        a_res = v_res
        all_l_res, all_a_res = [], []
        for _ in range(self.iter_size):
            tokens = torch.softmax(a_res['logits'], dim=-1)
            lengths = a_res['pt_lengths']
            lengths.clamp_(2, self.language.max_length)  # TODO:move to langauge model
            l_res = self.language(tokens, lengths)
            all_l_res.append(l_res)
            a_res = self.alignment(l_res['feature'], v_res['feature'])
            all_a_res.append(a_res)
        if self.training:
            return all_a_res, all_l_res, v_res
        else:
            return a_res, all_l_res[-1], v_res


class ABINet(nn.Module):
    def __init__(
        self, max_label_length: int, num_classes: int, pad_id: int, bos_id: int, eos_id: int, weight_decay: float = 0.0,
        iter_size: int = 3, d_model: int = 512, d_inner: int = 2048,
        dropout: float = 0.1, activation: str = 'relu', nhead: int = 8,
        v_loss_weight: float = 1.0, v_attention: str = 'position', v_attention_mode: str = 'nearest', 
        v_backbone: str = 'transformer', v_num_layers: int = 3,
        l_loss_weight: float = 1.0, l_num_layers: int = 4, l_detach: bool = True, l_use_self_attn: bool = False,
        a_loss_weight: float = 1.0, lm_only: bool = False, **kwargs
    ):
        super().__init__()
        self.weight_decay = weight_decay
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_label_length = max_label_length
        self.num_classes = num_classes

        self.model = ABINetIterModel(
            max_label_length, self.eos_id, num_classes, iter_size,
            d_model, nhead, d_inner, dropout, activation,
            v_loss_weight, v_attention, v_attention_mode, v_backbone, v_num_layers,
            l_loss_weight, l_num_layers, l_detach, l_use_self_attn, a_loss_weight
        )
        self.model.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'model.language.proj.weight'}

    def _add_weight_decay(self, model: nn.Module, skip_list=()):
        if self.weight_decay:
            return param_groups_weight_decay(model, self.weight_decay, skip_list)
        else:
            return [{'params': model.parameters()}]

    def learnable_params(self):
        params = []
        params.extend(self._add_weight_decay(self.model.vision))
        params.extend(self._add_weight_decay(self.model.alignment))
        # We use a different learning rate for the LM.
        for p in self._add_weight_decay(self.model.language, ('proj.weight',)):
            params.append(p)
        return params

    def forward(self, images: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        logits = self.model.forward(images)[0]['logits']
        return logits[:, :max_length + 1]  # truncate


    def _prepare_inputs_and_targets(self, labels):
        # Use dummy label to ensure sequence length is constant.
        dummy = ['0' * self.max_label_length]
        targets = self.tokenizer.encode(dummy + list(labels), self.device)[1:]
        targets = targets[:, 1:]  # remove <bos>. Unused here.
        # Inputs are padded with eos_id
        inputs = torch.where(targets == self.pad_id, self.eos_id, targets)
        inputs = F.one_hot(inputs, self.num_classes).float()
        lengths = torch.as_tensor(list(map(len, labels)), device=self.device) + 1  # +1 for eos
        return inputs, lengths, targets




def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)