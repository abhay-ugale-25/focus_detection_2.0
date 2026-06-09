import torch
import torch.nn as nn
import torch.optim as optim
from data_prep import get_dataloaders


# --- Model Architecture ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_length, input_size)
        out, _ = self.lstm(x)
        # Take hidden state of the last time step
        out = out[:, -1, :]
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


def train():
    # --- Configuration ---
    CSV_PATH = "lstm_training_data.csv"
    SEQ_LENGTH = 30
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    EARLY_STOP_PATIENCE = 5
    MODEL_SAVE_PATH = "focus_lstm_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Data ---
    train_loader, val_loader = get_dataloaders(
        CSV_PATH, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE
    )
    print()

    # --- Model, Loss, Optimizer ---
    model = LSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop with Early Stopping ---
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        # -- Training phase --
        model.train()
        train_loss = 0.0
        train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # -- Validation phase --
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                val_batches += 1

                _, predicted = torch.max(outputs, dim=1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        avg_val_loss = val_loss / val_batches
        val_accuracy = 100.0 * correct / total

        # -- Early Stopping Check --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            marker = "  ✓ saved"
        else:
            epochs_without_improvement += 1
            marker = ""

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}]  "
            f"Train Loss: {avg_train_loss:.4f}  |  "
            f"Val Loss: {avg_val_loss:.4f}  |  "
            f"Val Acc: {val_accuracy:.2f}%{marker}"
        )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    # --- Report ---
    print(f"\nModel saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
