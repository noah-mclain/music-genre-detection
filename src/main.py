import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from database import GTZANDataset
from models import CNNLSTMAttentionModel


def main():
    # Hyperparameters
    data_dir = ""
    batch_size = 8
    num_epochs = 4
    learning_rate = 1e-3
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    n_mels = 128

    # prepare Dataset and Dataloaders
    train_dataset = GTZANDataset(f"{data_dir}/train", n_mels=n_mels)
    val_dataset = GTZANDataset(f"{data_dir}/val", n_mels=n_mels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.genres)

    # Initialize model, loss, and optimizer
    model = CNNLSTMAttentionModel(num_genres=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # Save model
    torch.save(model.state_dict(), "cnn_lstm_attention.pth")
    print("Model saved as cnn_lstm_attention.pth")


if __name__ == "__main__":
    main()
