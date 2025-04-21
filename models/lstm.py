import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class BTimeSeriesLSTMClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes, seq_length, class_weights, hidden_size=128, dropout=0.3):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Optional embedding layer to project input channels
        self.embedding = nn.Linear(input_dim, hidden_size)

        # Two-layer LSTM (can make it bidirectional easily)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Dropout before classification
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [B, T, C]
        x = self.embedding(x)             # [B, T, hidden]
        lstm_out, _ = self.lstm(x)        # [B, T, hidden]
        last_hidden = lstm_out[:, -1, :]  # Take the last time step
        out = self.dropout(last_hidden)
        return self.fc(out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=-1)
        self.log("val_loss", loss, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": y}

    def on_validation_epoch_end(self):
        pass  # Add metrics here if needed

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]
