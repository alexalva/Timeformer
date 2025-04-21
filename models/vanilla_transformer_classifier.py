import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class TimeSeriesTransformerClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes, seq_length, class_weights, num_heads=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        #self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)

        self.positional_encoding = self.create_positional_encoding(seq_length, input_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def create_positional_encoding(self, seq_len, dim):
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])  # truncate if odd
        return pe

    def forward(self, x):
        x = x + self.positional_encoding.to(x.device)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]
