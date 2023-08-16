import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .ctc_decoder import ctc_decode
from torchmetrics.text import CharErrorRate

class CTCBaseline(pl.LightningModule):
    """
    CTC Baseline model for training
    """
    def __init__(self, model: nn.Module, args=None):
        """
        Arguments:
        ----------
        model: nn.Module
            An image2seq model

        args:
            Arguments for training
        """
        super().__init__()
        self.model = model
        self.args = args
        self.example_input_array = torch.Tensor(64, 3, 32, 128)
        self.cer = CharErrorRate()

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        # training step
        criterion = nn.CTCLoss(zero_infinity=True)
        images, targets, target_lengths = batch
        batch_size = images.size(0)
        target_lengths = torch.flatten(target_lengths)

        # Compute loss
        self.model.train()
        logits = self.model(images)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        log_probs = F.log_softmax(logits, dim=2)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        self.log('train_loss', loss, reduce_fx='mean', prog_bar=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        criterion = nn.CTCLoss(zero_infinity=True)
        images, targets, target_lengths = batch
        batch_size = images.size(0)
        target_lengths = torch.flatten(target_lengths)

        # Compute loss
        self.model.eval()
        logits = self.model(images)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        log_probs = F.log_softmax(logits, dim=2)
        val_loss = criterion(log_probs, targets, input_lengths, target_lengths)
        self.log('val_loss', val_loss, reduce_fx='mean', prog_bar=True)

        # Compute CER
        preds = ctc_decode(log_probs, method=self.args.decode_method, beam_size=self.args.beam_size)
        reals = targets.cpu().numpy().tolist()
        target_lengths = target_lengths.cpu().numpy().tolist()
        target_length_counter = 0
        total_cer = 0
        for pred, target_length in zip(preds, target_lengths):
            real = reals[target_length_counter:target_length_counter + target_length]
            target_length_counter += target_length
            total_cer += self.cer(torch.LongTensor(pred).view(1, -1), torch.LongTensor(real).view(1, -1))
        self.log('val_cer', total_cer / batch_size, reduce_fx='mean', prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        return optimizer