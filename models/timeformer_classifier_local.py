import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class_weights should be a tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Move class weights to the same device as inputs
        alpha = self.alpha.to(inputs.device) if self.alpha is not None else None

        # Compute cross-entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=alpha)

        pt = torch.exp(-ce_loss)  # probability of the correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class TransformerEncoderLayerWithTemperature(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Define activation explicitly here (standard choice: ReLU or GELU)
        self.activation = nn.ReLU()

        # Adaptive temperature parameter
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # You can ignore the masks unless you plan to use them
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )

        attn_weights = F.softmax(attn_weights / self.temperature.clamp(min=1e-2), dim=-1)
        attn_output = torch.bmm(attn_weights, src)

        # follow with norm + ff layers like standard transformer
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src
    
class TransformerEncoderLayerWithAttnOut(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.max_len = max_len
        self.batch_first = batch_first

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # === QKV projection ===
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=True)
        self.out_proj = nn.Linear(d_model, d_model)

        # === Relative Positional Bias ===
        self.relative_position_bias = nn.Parameter(
            torch.zeros(nhead, 2 * max_len - 1)
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

        # === Learnable Temperature ===
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # === Feed-forward ===
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.activation = nn.GELU()
        self.ff_scale = nn.Parameter(torch.ones(1))

        # === Normalization and Dropout ===
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def _get_rel_pos_bias(self, seq_len):
        needed_size = 2 * seq_len - 1
        if needed_size > self.relative_position_bias.shape[1]:
            # Expand bias table if it's too small
            new_bias = torch.zeros(self.nhead, needed_size, device=self.relative_position_bias.device)
            new_bias[:, :self.relative_position_bias.shape[1]] = self.relative_position_bias
            self.relative_position_bias = nn.Parameter(new_bias)

        pos = torch.arange(seq_len, device=self.relative_position_bias.device)
        rel_pos = pos[None, :] - pos[:, None] + seq_len - 1
        return self.relative_position_bias[:, rel_pos]


    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.batch_first:
            B, T, C = src.shape
        else:
            T, B, C = src.shape
            src = src.transpose(0, 1)  # make it batch-first for simplicity

        # === QKV projection ===
        qkv = self.qkv_proj(src)  # [B, T, 3C]
        qkv = qkv.reshape(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, h, T, d]

        # === Attention logits ===
        logits = (q @ k.transpose(-2, -1)) / self.temperature.clamp(min=1e-2)
        logits += self._get_rel_pos_bias(T).unsqueeze(0)  # [1, h, T, T]

        if src_mask is not None:
            logits = logits.masked_fill(src_mask == 0, float("-inf"))

        attn_weights = F.softmax(logits, dim=-1)
        attn_output = (attn_weights @ v)  # [B, h, T, d]
        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)
        attn_output = self.out_proj(attn_output)

        # === Residual + Norm ===
        src = self.norm1(src + self.dropout1(attn_output))

        # === Feed-forward ===
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(self.ff_scale * ff))

        return src, attn_weights  # attn_weights: [B, h, T, T]





class TimeSeriesTransformerClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes, seq_length, class_weights, num_heads_temporal=1, num_layers_temporal=2, num_heads_channel = 8, num_layers_channels = 2, dropout=0.5):
        super().__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        # self.loss_fn = FocalLoss(alpha=class_weights, gamma=1.0)

        self.validation_outputs = []  # store outputs here

          # === Conv1D Frontend ===
        # self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1)

        # Batch Normalization and ReLU activation
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU()
        )



        # Positional encoding for time-wise branch
        # self.positional_encoding_time = self.create_positional_encoding(seq_length, input_dim) #static positional encoding
        # self.positional_encoding_time = nn.Parameter(torch.randn(seq_length, input_dim))  # Shape: [T, C]

        self.contextual_pos_enc = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=5, padding=2, groups=input_dim)


        # Learnable Temperature Parameters
        self.temp_temperature = nn.Parameter(torch.tensor(1.0))
        self.channel_temperature = nn.Parameter(torch.tensor(1.0))

        
        # Temporal transformer (modeling across time)
        # temporal_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads_temporal, dropout=dropout, batch_first=True ) #attn_dropout=0.2

        # Learnable Temperature Parameters Temporal transformer (modeling across time)
        # temporal_layer = TransformerEncoderLayerWithTemperature(
        #     d_model=input_dim, nhead=num_heads, dropout=dropout, batch_first=True
        # )

        # self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=num_layers_temporal)

        self.temporal_layers = nn.ModuleList([
            TransformerEncoderLayerWithAttnOut(
                d_model=input_dim, nhead=num_heads_temporal, dropout=dropout, batch_first=True
            ) for _ in range(num_layers_temporal)
        ])


        # Channel-wise transformer (modeling across channels)
        # channel_layer = nn.TransformerEncoderLayer(d_model=seq_length, nhead=num_heads_channel, dropout=dropout, batch_first=True)

        # OR
        # Learnable Temperature Parameters Channel-wise transformer (modeling across channels)
        # channel_layer = TransformerEncoderLayerWithTemperature(
        #     d_model=seq_length, nhead=8, dropout=dropout, batch_first=True
        # )


        # self.channel_transformer = nn.TransformerEncoder(channel_layer, num_layers=num_layers_channels)  # improve num_layers 

        self.channel_layers = nn.ModuleList([
            TransformerEncoderLayerWithAttnOut(
                d_model=seq_length, nhead=num_heads_channel, dropout=dropout, batch_first=True
            ) for _ in range(num_layers_channels)
        ])


        # # Gating mechanism (learn to weigh contributions from each branch)
        # self.gate = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),
        #     nn.Sigmoid(),
        #     nn.Dropout(dropout)
        # )

        

        # # Final classification layer
        # self.fc = nn.Linear(input_dim, num_classes)

        # Projection layers before softmax gate
        self.fc_time = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.fc_channel = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )

        # Softmax-based gating
        self.gate_linear = nn.Linear(input_dim * 2, 2)

        # # === Dropout before classification ===
        self.dropout = nn.Dropout(dropout)

        # Final classifier after fusion
        self.output_layer = nn.Linear(input_dim * 2, num_classes)


    def create_positional_encoding(self, seq_len, dim):
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])  # handles odd dims
        return pe

    def forward(self, x, return_attn=False):
        # x shape: [B, T, C]
        batch_size = x.size(0)

        # === Conv1D Frontend ===
        x_conv = x.permute(0, 2, 1)              # [B, C, T]
        x_conv = self.conv1d(x_conv)             # [B, C, T]
        x = x_conv.permute(0, 2, 1)              # [B, T, C]

        # === Context-Aware Positional Encoding ===
        x_pe = x.permute(0, 2, 1)                # [B, C, T]
        x_pe = self.contextual_pos_enc(x_pe)     # [B, C, T]
        x_pe = x_pe.permute(0, 2, 1)             # [B, T, C]
        x_time = x + x_pe                        # Add positional context

        # === Temporal Transformer ===
        temporal_attn_weights = []
        for layer in self.temporal_layers:
            x_time, attn = layer(x_time)
            temporal_attn_weights.append(attn)   # [B, nhead, T, T]
        x_time_pooled = x_time.mean(dim=1)       # [B, C]

        # === Channel-wise Transformer ===
        x_channel = x.permute(0, 2, 1)           # [B, C, T]
        channel_attn_weights = []
        for layer in self.channel_layers:
            x_channel, attn = layer(x_channel)
            channel_attn_weights.append(attn)    # [B, nhead, C, C]
        x_channel_pooled = x_channel.mean(dim=2) # [B, C]

        # === Gated Fusion ===
        C = self.fc_time(x_time_pooled)          # [B, C]
        S = self.fc_channel(x_channel_pooled)    # [B, C]
        h = self.gate_linear(torch.cat([C, S], dim=-1))  # [B, 2]
        gates = F.softmax(h, dim=-1)             # [B, 2]
        g1, g2 = gates[:, 0].unsqueeze(1), gates[:, 1].unsqueeze(1)

        fused = torch.cat([C * g1, S * g2], dim=-1)  # [B, 2C]
        fused = self.dropout(fused)

        if return_attn:
            return self.output_layer(fused), temporal_attn_weights, channel_attn_weights
        else:
            return self.output_layer(fused)  # [B, num_classes]


    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=-1)
        
        self.validation_outputs.append({'loss': loss, 'preds': preds, 'targets': y})
        
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat([o['preds'] for o in self.validation_outputs])
        targets = torch.cat([o['targets'] for o in self.validation_outputs])
        val_loss = torch.stack([o['loss'] for o in self.validation_outputs]).mean()

        balanced_acc = balanced_accuracy_score(targets.cpu(), preds.cpu())
        f1 = f1_score(targets.cpu(), preds.cpu(), average='weighted')

        self.log_dict({
            'val_loss': val_loss,
            'balanced_acc': balanced_acc,
            'f1': f1
        }, prog_bar=True)

        # Clear for next epoch
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]
