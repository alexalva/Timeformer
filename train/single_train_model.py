import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.timeformer_classifier import TimeSeriesTransformerClassifier
from models.lstm import BTimeSeriesLSTMClassifier

from data.load_data import load_dataset
import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


# Hyperparameters
nhead_temporal = 16
nlayers_temporal = 2
nhead_channel = 8
nlayers_channel = 2
batch_size = 1

# Load data
dataset = "Libras"  # "BasicMotions" "HandMovementDirection"
extract_path = "./Multivariate_ts"
(X_train, y_train), (X_test, y_test), train_loader, test_loader, label_encoder = load_dataset(dataset, extract_path)

# Compute class weights
class_counts = np.bincount(y_train.numpy())
class_weights = torch.tensor((1.0 / (class_counts + 1e-5)), dtype=torch.float32)
# class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
class_weights /= class_weights.sum()

### Padding ###
# Check if input_dim is a multiple of 4

# Padding for channels (input_dim)
original_input_dim = X_train.shape[2]
padded_input_dim = math.ceil(original_input_dim / nhead_temporal) * nhead_temporal  # Ensure it's a multiple of 8
channel_padding_needed = padded_input_dim - original_input_dim

# Padding for sequence length (seq_length)
original_seq_length = X_train.shape[1]

padded_seq_length = math.ceil(original_seq_length / nhead_channel) * nhead_channel
seq_padding_needed = padded_seq_length - original_seq_length

if channel_padding_needed < 0 or seq_padding_needed < 0:
    raise ValueError("Padding is negative. Check your logic.")

# Apply padding
if channel_padding_needed > 0 or seq_padding_needed > 0:
    print(f"Padding features: {original_input_dim}→{padded_input_dim}, sequence: {original_seq_length}→{padded_seq_length}")
    X_train = F.pad(X_train, (0, channel_padding_needed, 0, seq_padding_needed), "constant", 0)
    X_test = F.pad(X_test, (0, channel_padding_needed, 0, seq_padding_needed), "constant", 0)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)


# Model setup
model = TimeSeriesTransformerClassifier(
    input_dim=padded_input_dim,
    num_classes=len(label_encoder.classes_),
    seq_length=X_train.shape[1],
    class_weights=class_weights,
    num_heads_temporal=nhead_temporal,
    num_layers_temporal=nlayers_temporal,
    num_heads_channel=nhead_channel,
    num_layers_channels=nlayers_channel,
    dropout=0.2
)
## Benchmark LSTM model
# model = BTimeSeriesLSTMClassifier(
#     input_dim=padded_input_dim,
#     num_classes=len(label_encoder.classes_),
#     seq_length=X_train.shape[1],
#     class_weights=class_weights
# )

early_stop_callback = EarlyStopping(
    monitor='val_loss',        # Metric to monitor
    patience=10,               # Stop if val_loss doesn't improve for 10 epochs
    mode='min',                # Because lower val_loss is better
    verbose=True
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',        # Save the model with the lowest val_loss
    save_top_k=1,              # Keep only the best model
    mode='min',
    filename='{epoch:02d}-{val_loss:.2f}'  # Optional: filename format
)


# Train
trainer = pl.Trainer(max_epochs=300, accelerator="auto", logger=False, callbacks=[early_stop_callback, checkpoint_callback])
trainer.fit(model, train_loader, test_loader)

# Load best checkpoint before evaluation
best_model_path = checkpoint_callback.best_model_path
model = model.__class__.load_from_checkpoint(best_model_path)


# Evaluate
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits = model(X_batch)
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
torch.save(model.state_dict(), f"{dataset}_transformer_model.pth")
