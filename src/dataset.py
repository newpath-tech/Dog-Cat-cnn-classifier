import os
import shutil
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_transforms(image_size: int = 128):
    """
    Data augmentation for training to prevent overfitting,
    and standard normalization for evaluation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


def build_cat_dog_subsets(data_root: str = "data", samples_per_class: int = 1000):
    """
    Prepares a clean dog vs cat dataset split if not already present.
    Extracts cat (label 3) and dog (label 5) images from torchvision standard set
    and organizes them into train/val/test directories.
    """
    root_path = Path(data_root)
    train_dir = root_path / "train"
    val_dir = root_path / "val"
    test_dir = root_path / "test"

    # If directories already have images, skip re-generation
    if (train_dir / "cats").exists() and len(list((train_dir / "cats").glob("*.jpg"))) > 50:
        print(f"Dataset already configured at {data_root}")
        return

    print("Setting up Cats vs Dogs dataset structure...")
    for split in ["train", "val", "test"]:
        for cls in ["cats", "dogs"]:
            os.makedirs(root_path / split / cls, exist_ok=True)

    # Download source images via torchvision
    import torchvision
    temp_dir = "./temp_cifar"
    train_src = torchvision.datasets.CIFAR10(root=temp_dir, train=True, download=True)
    test_src = torchvision.datasets.CIFAR10(root=temp_dir, train=False, download=True)

    # CIFAR-10 class indices: 3: 'cat', 5: 'dog'
    cat_idx, dog_idx = 3, 5

    cat_train_count = 0
    dog_train_count = 0
    cat_val_count = 0
    dog_val_count = 0

    val_threshold = int(samples_per_class * 0.2)

    for idx, (img, label) in enumerate(train_src):
        if label == cat_idx and (cat_train_count + cat_val_count) < samples_per_class:
            if cat_val_count < val_threshold:
                img.save(val_dir / "cats" / f"cat_{cat_val_count:04d}.jpg")
                cat_val_count += 1
            else:
                img.save(train_dir / "cats" / f"cat_{cat_train_count:04d}.jpg")
                cat_train_count += 1
        elif label == dog_idx and (dog_train_count + dog_val_count) < samples_per_class:
            if dog_val_count < val_threshold:
                img.save(val_dir / "dogs" / f"dog_{dog_val_count:04d}.jpg")
                dog_val_count += 1
            else:
                img.save(train_dir / "dogs" / f"dog_{dog_train_count:04d}.jpg")
                dog_train_count += 1

    # Test set
    test_cat_count = 0
    test_dog_count = 0
    test_limit = 200
    for idx, (img, label) in enumerate(test_src):
        if label == cat_idx and test_cat_count < test_limit:
            img.save(test_dir / "cats" / f"cat_test_{test_cat_count:04d}.jpg")
            test_cat_count += 1
        elif label == dog_idx and test_dog_count < test_limit:
            img.save(test_dir / "dogs" / f"dog_test_{test_dog_count:04d}.jpg")
            test_dog_count += 1

    # Also save dedicated standalone samples for easy inference tests
    sample_dir = root_path / "sample_test"
    os.makedirs(sample_dir, exist_ok=True)
    for i in range(min(5, test_cat_count)):
        shutil.copy(test_dir / "cats" / f"cat_test_{i:04d}.jpg", sample_dir / f"sample_cat_{i+1}.jpg")
    for i in range(min(5, test_dog_count)):
        shutil.copy(test_dir / "dogs" / f"dog_test_{i:04d}.jpg", sample_dir / f"sample_dog_{i+1}.jpg")

    print(f"Data preparation complete:")
    print(f"  Train: {cat_train_count} cats, {dog_train_count} dogs")
    print(f"  Validation: {cat_val_count} cats, {dog_val_count} dogs")
    print(f"  Test: {test_cat_count} cats, {test_dog_count} dogs")


def get_dataloaders(data_dir: str = "data", batch_size: int = 32, image_size: int = 128, num_workers: int = 0):
    train_transform, eval_transform = get_transforms(image_size=image_size)

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset.classes
