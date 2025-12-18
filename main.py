import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from src.database import GTZANDataset
from src.logging_config import setup_logging
from src.models import CNNLSTMAttentionModel, SimpleModel
from src.model_comparison import ModelComparison

logger = setup_logging()


def get_optimal_num_workers():
    try:
        import platform

        os_type = platform.system()
    except ImportError:
        os_type = "Unknown"

    if os_type == "windows":
        num_workers = 0
        logger.info("Detected Windows - using num_workers = 0")
    elif os_type == "Darwin":
        num_workers = 4
        logger.info("Detected macOS - using num_workers = 4")
    else:
        num_workers = 4
        logger.info("Detected Linux/Other - using num_workers = 4")
    return num_workers


def main():
    # Hyperparameters
    data_dir = "src/Data/genres_original"
    cache_dir = "src/preprocessed_cache"
    batch_size = 64
    num_epochs = 150
    learning_rate = 1e-2
    weight_decay = 5e-6
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    n_mels = 128

    # prepare Dataset and Dataloaders
    logger.info("\n" + "-" * 70)
    logger.info("Preparing Dataset with caching...")
    logger.info("-" * 70)
    logger.info("First run: Preprocesses dataset (~5-10 minutes)")
    logger.info("Subsequent runs: Loads from cache instantly")
    logger.info("-" * 70)

    dataset = GTZANDataset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        sr=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        duration=30.0,
        segment_length=130,
        augment=False,
        preprocess_all=True,  # Triggers preprocessing on first run
    )
    training_dataset = GTZANDataset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        sr=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        duration=30.0,
        segment_length=130,
        augment=True,
        preprocess_all=False,
    )
    val_dataset = GTZANDataset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        sr=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        duration=30.0,
        segment_length=130,
        augment=False,
        preprocess_all=False,
    )

    logger.info("\n" + "Dataset loaded successfully")
    dataset_info = dataset.get_dataset_info()
    logger.info(f"Total samples: {dataset_info['total_samples']}")
    logger.info(f"Number of genres: {dataset_info['num_genres']}")
    logger.info(f"Genres: {', '.join(dataset_info['genres'])}")
    logger.info(f"Sample rate: {dataset_info['sample_rate']} Hz")
    logger.info(f"Mel-bins: {dataset_info['n_mels']}")
    logger.info(f"FFT size: {dataset_info['n_fft']}")
    logger.info(f"Duration: {dataset_info['duration_seconds']} seconds")
    logger.info(f"Segment length: {dataset_info['segment_length']}")
    logger.info(f"Augmentation: {dataset_info['augmentation']}")
    logger.info(f"Caching: {dataset_info['caching']}")

    logger.info("\nGenre Distribution:")
    for genre, count in dataset_info["genre_distribution"].items():
        logger.info(f"  - {genre}: {count} samples")

    num_workers = get_optimal_num_workers()

    all_indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

    train_dataset = Subset(training_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    logger.info("\n" + "Dataset split successfully: 80% train, 20% validation")
    logger.info(f"Data augmentation enabled for training set: {train_dataset.dataset.augment}")
    logger.info(f"Data augmentation disabled for validation set: {val_dataset.dataset.augment}")

    logger.info(f"Train set size: {len(train_dataset)} samples")
    logger.info(f"Validation set size: {len(val_dataset)} samples")
    # val_dataset = GTZANDataset(f"{data_dir}/val", n_mels=n_mels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Training batches per epoch: {len(train_loader)}")
    logger.info(f"Validation batches per epoch: {len(val_loader)}")

    num_classes = len(dataset.genres)
    logger.info(f"Number of classes: {num_classes}")

    # Initialize model comparison
    comparison = ModelComparison(device, results_dir="../results")

    # Simple Model
    simple_model = SimpleModel(num_genres=num_classes).to(device)
    comparison.train_model(
        simple_model,
        "SimpleModel",
        train_loader,
        val_loader,
        num_epochs,
        learning_rate,
        weight_decay,
        "models/simple_model.pth",
    )

    # Complex Model (CNNLSTM + Temporal Attention)
    complex_model = CNNLSTMAttentionModel(num_genres=num_classes).to(device)
    comparison.train_model(
        complex_model,
        "ComplexModel",
        train_loader,
        val_loader,
        num_epochs,
        learning_rate,
        weight_decay,
        "models/complex_model.pth",
        pretrained_path="models/cnn_lstm_attention.pth",
    )

    comparison.generate_comparison_report()
    results_file = comparison.save_results_json()

    # # Initialize model, loss, and optimizer
    # model = CNNLSTMAttentionModel(num_genres=num_classes)
    # try:
    #     model.load_state_dict(torch.load("models/cnn_lstm_attention.pth", weights_only=True))
    #     logger.info("Loaded pretrained CNNLSTMAttention weight from models/cnn_lstm_attention.pth")
    # except FileNotFoundError:
    #     logger.info("No pretrained model found, training from scratch")

    # # model = SimpleModel(num_genres=num_classes)
    # model.to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # logger.info(f"Model moved to device: {device}")

    # logger.info("\n" + "-" * 70)
    # logger.info(f"Starting Training for {num_epochs}...")
    # logger.info("-" * 70)

    # best_val_acc = 0.0
    # best_epoch = 0
    # patience = 30
    # patience_counter = 0

    # # Training loop
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     correct = 0
    #     total = 0

    #     for batch_idx, (inputs, labels) in enumerate(train_loader):
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         # Backward pass
    #         loss.backward()
    #         # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         optimizer.step()

    #         running_loss += loss.item() * inputs.size(0)
    #         _, predicted = torch.max(outputs, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    #         if (batch_idx + 1) % 10 == 0:
    #             batch_loss = running_loss / total
    #             batch_acc = correct / total
    #             logger.info(
    #                 f"Epoch {epoch+1:2d}/{num_epochs} | "
    #                 f"Batch {batch_idx + 1:3d}/{len(train_loader)} | "
    #                 f"Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}"
    #             )

    #     train_loss = running_loss / total
    #     train_acc = correct / total

    #     # Validation
    #     model.eval()
    #     val_loss = 0.0
    #     val_correct = 0
    #     val_total = 0

    #     with torch.no_grad():
    #         for inputs, labels in val_loader:
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)
    #             outputs = model(inputs)
    #             loss = criterion(outputs, labels)

    #             val_loss += loss.item() * inputs.size(0)
    #             _, predicted = torch.max(outputs, 1)
    #             val_total += labels.size(0)
    #             val_correct += (predicted == labels).sum().item()

    #     val_loss /= val_total
    #     val_acc = val_correct / val_total
    #     logger.debug(f"Batch labels distribution: {torch.bincount(labels)}")

    #     scheduler.step(val_loss)
    #     current_lr = optimizer.param_groups[0]['lr']

    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         best_epoch = epoch + 1
    #         patience_counter = 0
    #         torch.save(model.state_dict(), "models/cnn_lstm_attention_best.pth")
    #         logger.debug(f"Saved best model at epoch {epoch+1} with val acc {best_val_acc:.4f}")
    #     else:
    #         patience_counter += 1
    #         logger.debug(
    #             f"No improvement in val acc for {patience_counter} epochs, {patience_counter}/{patience}"
    #         )

    #     if patience_counter >= patience:
    #         logger.info(f"Early stopping triggered at epoch {epoch+1}")
    #         logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    #         break

    #     logger.info(
    #         f"Epoch {epoch+1}/{num_epochs} | "
    #         f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
    #         f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
    #         f"LR: {current_lr:.2e}"  # Show current learning rate
    #     )
    # logger.info("n" + "-" * 70)

    # # Save model
    # torch.save(model.state_dict(), "models/cnn_lstm_attention.pth")
    # logger.info("Model saved as models/cnn_lstm_attention.pth")
    # logger.info("Best model saved as models/cnn_lstm_attention_best.pth")
    # logger.info(f"   Best validation accuracy: {best_val_acc:.4f}")

    logger.info("n" + "-" * 70)
    logger.info("Training complete")
    logger.info(f"Comparison results saved to: {results_file}")
    logger.info("n" + "-" * 70)


if __name__ == "__main__":
    main()
