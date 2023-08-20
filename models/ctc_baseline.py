import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from .ctc_decoder import ctc_decode
from .utils import rule
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
        self.example_input_array = torch.Tensor(64, 3, args.height, args.width)
        self.cer = CharErrorRate()
        self.automatic_optimization = False


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
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
        criterion = nn.CTCLoss(zero_infinity=True)
        images, targets, target_lengths = batch
        batch_size = images.size(0)
        target_lengths = torch.flatten(target_lengths)

        # Compute loss
        self.model.eval()
        with torch.no_grad():
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
        print(f'val_loss: {val_loss}. val_cer: {total_cer / batch_size}', end='\r')


    def configure_optimizers(self):
        optimizer_params = {
            'lr': self.args.lr,
            'weight_decay': self.args.weight_decay
        }

        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), betas=(self.args.momentum, 0.999), **optimizer_params
            )
        elif self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), momentum=self.args.momentum, **optimizer_params
            )
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), betas=(self.args.momentum, 0.999), **optimizer_params
            )
        elif self.args.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(
                self.parameters(), **optimizer_params, eps=1e-8, rho=0.95
            )

        # Linear warmup
        if self.args.warmup_steps > 0:
            # Linear scheduler
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, rule(self.args)
            )
            return [optimizer, ], [scheduler, ]

        return optimizer