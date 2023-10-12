import torch
import torch.nn as nn
import timm

from x_transformers import *
from x_transformers.autoregressive_wrapper import *


class SwinTransformerOCR(nn.Module):
    def __init__(self, converter, temperature=0.2):
        super().__init__()
        self.converter = converter

        self.encoder = timm.create_model(
            'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
            pretrained=True,
            features_only=True,
            out_indices=(2,)
        )
        self.decoder = CustomARWrapper(
            TransformerWrapper(
                num_tokens=converter.num_classes,
                max_seq_len=converter.batch_max_length,
                attn_layers=Decoder(
                    dim=384, depth=4, heads=8,
                    cross_attend=True, ff_glu=False, attn_on_attn=False,
                    use_scalenorm=False, rel_pos_bias=False
                )
            )
        )
        self.temperature = temperature
        self.bos_token = converter.bos_id
        self.eos_token = converter.eos_id
        self.pad_token = converter.pad_id


    def forward(self, x):
        '''
        x: (B, C, W, H)
        labels: (B, S)

        # B : batch size
        # W : image width
        # H : image height
        # S : source sequence length
        # E : hidden size
        # V : vocab size
        '''

        encoded = self.encoder(x)[0]
        dec = self.decoder.generate(torch.LongTensor([self.bos_token]*len(x))[:, None].to(x.device), self.max_seq_len,
                                    eos_token=self.eos_token, context=encoded, temperature=self.temperature)
        return dec, encoded
    

    @torch.no_grad()
    def predict(self, image):
        dec, _ = self(image)
        pred = self.converter.decode(dec)
        return pred

    

class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *cfg, **kwcfg):
        super(CustomARWrapper, self).__init__(*cfg, **kwcfg)

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwcfg):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwcfg.pop('mask', None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]

            logits = self.net(x, mask=mask, **kwcfg)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
            else:
                raise NotImplementedError

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out