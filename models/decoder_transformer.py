import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from .transformer import Transformer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x)


class TransformerRecognitionHead(nn.Module):
    """
    input: [b x T x in_planes]
    output: probability sequence: [b x T x num_classes]
    """
    def __init__(self, num_classes, in_planes, sDim=512, max_len_labels=25, encoder_block=4, decoder_block=4):
        super(TransformerRecognitionHead, self).__init__()
        self.num_classes = num_classes # this is the output classes. So it includes the <EOS>.
        self.in_planes = in_planes
        self.sDim = sDim
        self.max_len_labels = max_len_labels + 1 # +1 for <BOS>
        self.decoder = DecoderUnit(sDim=sDim, xDim=in_planes, yDim=num_classes, num_encoders=encoder_block, num_decoders=decoder_block)
  

    def forward(self, x, text=None, is_train=True, tgt_padding_mask=None):
        # Decoder
        if is_train:
            tgt_mask = self._generate_square_subsequent_mask(text.size(1))
            outputs = self.decoder(x, text, tgt_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)
            return outputs  # (N, T, C)
        else:
            batch_size = x.size(0)
            # Decoder
            predicted_scores = torch.zeros(batch_size, self.max_len_labels - 1, self.num_classes)
            outputs = torch.zeros(batch_size, self.max_len_labels).long()  # Initialize prediction
            # Initialize padding mask to true
            tgt_padding_mask = torch.BoolTensor(batch_size, self.max_len_labels).fill_(True)
            for i in range(self.max_len_labels - 1):
                tgt_padding_mask[:, i] = False
                pred_prob = self.decoder(x, outputs, tgt_padding_mask=tgt_padding_mask)  # (N, T, C)
                pred_prob = pred_prob[:, i, :]  # (N, C)
                predicted_scores[:, i, :] = pred_prob
                _, preds_index = pred_prob.max(-1)  # (N)
                outputs[:, i + 1] = preds_index

            return predicted_scores
        
    def _generate_square_subsequent_mask(self, x):
        mask = (torch.triu(torch.ones(x, x)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



class DecoderUnit(nn.Module):
    def __init__(self, sDim, xDim, yDim, num_encoders, num_decoders):
        super(DecoderUnit, self).__init__()
        self.sDim = sDim
        self.xDim = xDim
        self.yDim = yDim
        self.emdDim = sDim
    
        self.tgt_embedding = nn.Embedding(yDim, self.emdDim) # the last is used for <BOS> 
        self.src_embedding = nn.Linear(xDim, sDim)

        self.fc = nn.Linear(sDim, yDim)
        self.decoder = Transformer(d_model=sDim, nhead=8, num_encoder_layers=num_encoders, num_decoder_layers=num_decoders, dim_feedforward=2048)
        self.pos = PositionalEncoding(self.emdDim, 0.1)
        self.init_weights()

    def init_weights(self):
        init.normal_(self.tgt_embedding.weight, std=0.01)
        init.normal_(self.fc.weight, std=0.01)
        init.constant_(self.fc.bias, 0)


    def forward(self, x, yPrev, tgt_padding_mask=None, tgt_mask=None):
        # x: feature sequence from the image decoder.
        # batch_size, T, _ = x.size()
        yProj = self.tgt_embedding(yPrev.long())
        yProj = yProj.transpose(1, 0)
        yProj = self.pos(yProj)

        x = self.src_embedding(x)
        x = x.transpose(1, 0)
        output = self.decoder(x, yProj, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        output = self.fc(output)
        output = output.transpose(1, 0)
        
        return output