import torch
import pandas as pd
import numpy as np
from statistics import mode
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def get_dataloaders(csv_path, seq_length=30, batch_size=32, train_split=0.8):
    # --- 1. Data Loading & Cleaning ---
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    # --- 2. Feature Scaling ---
    scaler = StandardScaler()
    features = df[["avg_ear", "avg_gaze", "pitch_ratio_delta"]].values
    df[["avg_ear", "avg_gaze", "pitch_ratio_delta"]] = scaler.fit_transform(features)

    # --- 3. Stratified Sequence Generation ---
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    # Group the dataframe by label
    for label, group in df.groupby("label"):
        group_features = group[["avg_ear", "avg_gaze", "pitch_ratio_delta"]].values
        group_labels = group["label"].values
        
        X_group, y_group = [], []
        
        # Generate 30-frame overlapping sequences
        for i in range(len(group_features) - seq_length):
            X_group.append(group_features[i : i + seq_length])
            window_labels = group_labels[i : i + seq_length]
            y_group.append(mode(window_labels))
            
        if len(X_group) == 0:
            continue
            
        X_group = np.array(X_group)
        y_group = np.array(y_group)
        
        # Calculate split index for this specific group
        split_idx = int(len(X_group) * train_split)
        
        # Append 80% to train, 20% to val
        X_train_list.append(X_group[:split_idx])
        y_train_list.append(y_group[:split_idx])
        
        X_val_list.append(X_group[split_idx:])
        y_val_list.append(y_group[split_idx:])

    # Combine all sequences
    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])
    X_val = np.concatenate(X_val_list, axis=0) if X_val_list else np.array([])
    y_val = np.concatenate(y_val_list, axis=0) if y_val_list else np.array([])

    # --- 4. Tensor Conversion & Shuffling ---
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Create random permutation index to shuffle training tensors
    perm_indices = torch.randperm(len(X_train_tensor))
    X_train_tensor = X_train_tensor[perm_indices]
    y_train_tensor = y_train_tensor[perm_indices]

    # --- 5. DataLoader Wrapping ---
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset loaded from: {csv_path}")
    print(f"  Train sequences:  {len(X_train_tensor)}")
    print(f"  Val sequences:    {len(X_val_tensor)}")
    print(f"  Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Class distribution (val):   {dict(zip(*np.unique(y_val, return_counts=True)))}")

    return train_loader, val_loader
