import argparse
import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from src.model import ThreeLayerCNN
from src.dataset import get_dataloaders, build_cat_dog_subsets


def train(data_dir: str = "data",
          epochs: int = 12,
          batch_size: int = 32,
          lr: float = 1e-3,
          weight_decay: float = 1e-4,
          save_path: str = "models/cat_dog_cnn3.pth"):
    
    # Ensure dataset exists
    build_cat_dog_subsets(data_root=data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=128
    )

    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_loader)} | Validation batches: {len(val_loader)}")

    model = ThreeLayerCNN(num_classes=len(classes), dropout_rate=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_acc = 0.0
    start_time = time.time()

    print("\nStarting training loop...")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels).item()
                total_val += labels.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        scheduler.step()

        saved_flag = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            saved_flag = " [Checkpoint Saved]"

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:5.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:5.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}{saved_flag}")

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (Saved to {save_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3-Layer CNN for Dog vs Cat Classification")
    parser.add_argument("--data-dir", type=str, default="data", help="Root dataset directory")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--save-path", type=str, default="models/cat_dog_cnn3.pth", help="Checkpoint save destination")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )
