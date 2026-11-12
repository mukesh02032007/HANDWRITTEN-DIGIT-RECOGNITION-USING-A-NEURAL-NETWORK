# Handwritten Character Recognition using CNN

A deep learning mini project that classifies 28×28 grayscale handwritten images into one of 26 classes (A–Z) using a Convolutional Neural Network built with Keras/TensorFlow.

## Features

- Data loading and preprocessing from a 785-column CSV (1 label + 784 pixels)
- Exploratory data analysis: class distribution chart and sample image grid
- CNN model with 3 convolution + max-pooling blocks
- Training, validation and model saving (`.h5`)
- Prediction on a custom image using OpenCV

## Tech Stack

Python, TensorFlow/Keras, OpenCV, NumPy, Pandas, Matplotlib, scikit-learn

## Dataset

`handwritten_data_785.csv`: each row is one image.

| Column | Description |
|--------|-------------|
| `0` | Label (0–25 → A–Z) |
| `1–784` | Flattened 28×28 pixel intensities |

> The dataset is not included in this repo because of its size. Download an A–Z handwritten dataset in this format and place the CSV in the project root.

## Model Architecture

| Layer | Output Shape |
|-------|--------------|
| Conv2D (32, 3×3) + MaxPool | 13×13×32 |
| Conv2D (64, 3×3, same) + MaxPool | 6×6×64 |
| Conv2D (128, 3×3, valid) + MaxPool | 2×2×128 |
| Flatten | 512 |
| Dense (64, ReLU) | 64 |
| Dense (128, ReLU) | 128 |
| Dense (26, Softmax) | 26 |

- Total params: 411,536 (137,178 trainable)
- Optimizer: Adam (lr = 0.001)
- Loss: Categorical Crossentropy

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 95.21% |
| Validation Accuracy | 97.26% |
| Training Loss | 0.1790 |
| Validation Loss | 0.1009 |

Trained for 1 epoch. Increase `epochs` for better results.

## Installation

```bash
git clone https://github.com/mukesh02032007/DL.git
cd DL
pip install numpy pandas matplotlib scikit-learn opencv-python tensorflow
```

## Usage

1. Place `handwritten_data_785.csv` in the project folder.
2. Place the image you want to predict as `test_image.jpg`.
3. Run:

```bash
python "Handwritten Digit Recognition.py"
```

The script trains the model, saves it as `model_hand.h5`, and opens a window showing the predicted character. Press `Esc` to close it.

## Project Structure

```
DL/
├── Handwritten Digit Recognition.py
├── handwritten_data_785.csv      # not included
├── test_image.jpg                # your input image
└── model_hand.h5                 # generated after training
```

## Future Improvements

- Train for more epochs with `EarlyStopping` and `ReduceLROnPlateau`
- Add data augmentation
- Add confusion matrix and per-class metrics
- Build a web interface for image upload and prediction

## References

- [TensorFlow](https://www.tensorflow.org/api_docs) · [Keras](https://keras.io/) · [OpenCV](https://docs.opencv.org/) · [scikit-learn](https://scikit-learn.org/stable/)

## Author

Mukesh
