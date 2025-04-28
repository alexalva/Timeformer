import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from aeon.datasets import load_classification

def df_list_to_3d_array(X_df):
    n_instances = len(X_df)
    seq_length = X_df[0].shape[0]
    n_channels = X_df[0].shape[1]
    X_np = np.zeros((n_instances, seq_length, n_channels))
    for i, ts in enumerate(X_df):
        X_np[i] = ts
    return X_np

def load_dataset(dataset_name, extract_path, batch_size=16):
    # Load TRAIN
    X_train_df, y_train, _ = load_classification(dataset_name, split="train", extract_path=extract_path, return_metadata=True)
    X_test_df, y_test, _ = load_classification(dataset_name, split="test", extract_path=extract_path, return_metadata=True)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    X_train_np = df_list_to_3d_array(X_train_df)
    X_test_np = df_list_to_3d_array(X_test_df)

    scaler = StandardScaler()
    X_train_flat = X_train_np.reshape(-1, X_train_np.shape[2])
    X_test_flat = X_test_np.reshape(-1, X_test_np.shape[2])
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train_np.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test_np.shape)

    # Torch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor), train_loader, test_loader, label_encoder

extract_path = "./Multivariate_ts"
dataset = "AtrialFibrillation"

(X_train, y_train), (X_test, y_test), train_loader, test_loader, label_encoder = load_dataset(dataset, extract_path)
