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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



def train_and_evaluate(dataset, run_id, model_fn):

    extract_path = "./Multivariate_ts"


    # Hyperparameters
    nhead_temporal = 16
    nlayers_temporal = 2
    nhead_channel = 8
    nlayers_channel = 2
    batch_size = 1
    max_epochs = 50


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
    # model = TimeSeriesTransformerClassifier(
    #     input_dim=padded_input_dim,
    #     num_classes=len(label_encoder.classes_),
    #     seq_length=X_train.shape[1],
    #     class_weights=class_weights,
    #     num_heads_temporal=nhead_temporal,
    #     num_layers_temporal=nlayers_temporal,
    #     num_heads_channel=nhead_channel,
    #     num_layers_channels=nlayers_channel,
    #     dropout=0.2
    # )
    ## Benchmark LSTM model
    # model = BTimeSeriesLSTMClassifier(
    #     input_dim=padded_input_dim,
    #     num_classes=len(label_encoder.classes_),
    #     seq_length=X_train.shape[1],
    #     class_weights=class_weights
    # )

    # === Model setup (via passed function) ===
    model = model_fn(
        input_dim=padded_input_dim,
        num_classes=len(label_encoder.classes_),
        seq_length=X_train.shape[1],
        class_weights=class_weights
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',        # Metric to monitor
        patience=30,               # Stop if val_loss doesn't improve for 10 epochs
        mode='min',                # Because lower val_loss is better
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',        # Save the model with the lowest val_loss
        save_top_k=1,              # Keep only the best model
        mode='min',
        filename='{epoch:02d}-{val_loss:.2f}'  # Optional: filename format
    )

    # Dynamically get model name from the class
    model_name = model.__class__.__name__

    # Define results dir with model name
    results_dir = f"./results/{dataset}/{model_name}/run_{run_id}"
    os.makedirs(results_dir, exist_ok=True)


    # Train
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", logger=False, callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model, train_loader, test_loader)


    # Evaluate
    # === Evaluation ===
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # === Save Classification Report ===
    report_dict = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True)
    with open(os.path.join(results_dir, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=4)

    # === Save Confusion Matrix Plot ===
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(xticks_rotation=45)
    plt.title(f"{dataset} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    # === Save Predictions & Labels (Optional) ===
    np.save(os.path.join(results_dir, "y_true.npy"), all_labels)
    np.save(os.path.join(results_dir, "y_pred.npy"), all_preds)

    # === Save the model (Optional) ===
    torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))

    # === Optional: Save Attention Maps (only for Transformer models) ===
    if isinstance(model, TimeSeriesTransformerClassifier):
        model.eval()
        with torch.no_grad():
            # Get a single example from the test set
            sample_batch = next(iter(test_loader))[0][:1].to(model.device)
            output, temporal_attn, channel_attn = model(sample_batch, return_attn=True)

            # === Plot Temporal Attention (Last Layer, Averaged Heads) ===
            temp_attn_avg = temporal_attn[-1].mean(dim=1)[0].cpu().numpy()  # shape [T, T]
            plt.figure(figsize=(6, 5))
            plt.imshow(temp_attn_avg, cmap="viridis")
            plt.title("Temporal Attention (Last Layer)")
            plt.xlabel("Time Step")
            plt.ylabel("Time Step")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "temporal_attention.png"))
            plt.close()

            # === Plot Channel Attention (Last Layer, Averaged Heads) ===
            chan_attn_avg = channel_attn[-1].mean(dim=1)[0].cpu().numpy()  # shape [C, C]
            plt.figure(figsize=(6, 5))
            plt.imshow(chan_attn_avg, cmap="plasma")
            plt.title("Channel Attention (Last Layer)")
            plt.xlabel("Channel")
            plt.ylabel("Channel")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "channel_attention.png"))
            plt.close()

    return {
        "dataset": dataset,
        "run_id": run_id,
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_acc": balanced_accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average='weighted'),
        "report_path": os.path.join(results_dir, "classification_report.json"),
        "conf_matrix_path": os.path.join(results_dir, "confusion_matrix.png")
    }