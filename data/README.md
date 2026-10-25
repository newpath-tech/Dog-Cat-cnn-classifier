# Dataset Organization

This project structures the image dataset into standard class directories for binary classification (`cats` vs `dogs`).

## Directory Layout
```
data/
├── train/
│   ├── cats/
│   └── dogs/
├── val/
│   ├── cats/
│   └── dogs/
├── test/
│   ├── cats/
│   └── dogs/
└── sample_test/
    ├── sample_cat_1.jpg
    ├── sample_dog_1.jpg
    └── ...
```

## Dataset Setup
1. The included Jupyter notebook and data loader script (`src/dataset.py`) automatically prepare the dataset splits on the first run.
2. Alternatively, if using the full Kaggle Dogs vs Cats dataset (25,000 images):
   - Place cat images into `data/train/cats/` and `data/val/cats/`.
   - Place dog images into `data/train/dogs/` and `data/val/dogs/`.
   - The PyTorch `ImageFolder` loader automatically scans and processes images from these folders.
