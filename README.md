# Cat vs Dog Image Classification using SVM

A **Support Vector Machine (SVM)** classifier that distinguishes between images of cats and dogs, built on a ~4,000-image subset of the Kaggle
["Dogs vs. Cats"](https://www.kaggle.com/c/dogs-vs-cats) dataset.

## Overview

| | |
|---|---|
| **Task** | Binary image classification (Cat vs Dog) |
| **Approach** | Classical ML — SVM with hand-crafted features (not deep learning) |
| **Model** | `sklearn.svm.SVC` (RBF kernel), tuned via `GridSearchCV` |
| **Dataset** | Kaggle Dogs vs. Cats (~4,000 images, ~2,000 per class) |

## Why HOG + Color Histograms Instead of Raw Pixels

A naive approach — resizing images and flattening raw pixel values directly into the SVM — caps out around **~70–72% accuracy**, barely better than guessing. Raw pixels mostly encode noise (lighting, background, pose) rather than the actual shape information that separates a cat from a dog.

This project instead extracts:
- **HOG (Histogram of Oriented Gradients)** — captures edge orientation and local shape structure (ear shape, face contours), which is what visually distinguishes the two classes.
- **Color histograms** — a compact 96-value summary of color distribution per image, as complementary signal HOG doesn't capture.

This combination gives the SVM meaningfully more informative features to learn from, without switching to a deep learning approach.

## Project Structure

```
├── train cat dog file/                    # Folder of cat.N.jpg / dog.N.jpg images (Google Drive)
├── Cat_Dog_SVM_Classification.ipynb       # Full notebook: EDA, feature extraction, training, evaluation
└── README.md
```

## Workflow

1. **Load dataset** — mount Google Drive, verify image counts per class.
2. **Preview samples** — visually inspect a few cat/dog images before processing.
3. **Feature extraction** — resize to 128×128, extract HOG features (grayscale) + normalized color histograms (per channel), concatenate into one feature vector per image.
4. **Feature scaling** — `StandardScaler`, since SVM is distance-based.
5. **Train/test split** — 80/20, stratified to preserve the 50/50 class balance.
6. **Hyperparameter tuning** — `GridSearchCV` over `C` and `gamma` (RBF kernel) instead of using default values.
7. **Evaluation** — accuracy, classification report, confusion matrix, 5-fold cross-validation.
8. **Prediction visualization** — display sample test images with actual vs predicted labels.

## Results

> Fill in with your actual run output from Colab:

| Metric | Score |
|---|---|
| Test Accuracy | _e.g. 0.XX_ |
| Average CV Accuracy | _e.g. 0.XX_ |
| Best hyperparameters | _e.g. C=__, gamma=___ |

## Limitations

This is a **classical ML** approach — HOG and color histograms are hand-crafted features, not learned representations. It will not match the accuracy of a Convolutional Neural Network (CNN), which learns its own features directly from raw pixels. This project's goal is to demonstrate SVM-based classification and feature engineering, not to reach state-of-the-art image classification accuracy.

## How to Run

1. Upload the cat/dog image folder to Google Drive (filenames like `cat.123.jpg`, `dog.456.jpg`).
2. Open `Cat_Dog_SVM_Classification.ipynb` in Google Colab.
3. Update `dataset_path` to point to your Drive folder.
4. Run all cells top to bottom (note: HOG extraction over ~4,000 images and `GridSearchCV` both take a few minutes).

## Requirements

```
opencv-python
numpy
matplotlib
seaborn
scikit-image
scikit-learn
```

Install with:
```bash
pip install opencv-python numpy matplotlib seaborn scikit-image scikit-learn
```
