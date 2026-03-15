# ASL Hand Sign Recognition using MediaPipe and Neural Networks
This project implements a hand sign recognition system for the American Sign Language (ASL) alphabet using hand keypoints extracted with MediaPipe and a neural network trained on these keypoints.

The pipeline consists of:

1. Extracting hand landmarks from images using MediaPipe
2. Converting landmarks into a numerical dataset
3. Training a neural network to classify ASL gestures

The model learns to recognize hand signs based on **21 hand landmarks (63 coordinates)**.

---

# Project Structure

```
ASL_MediaPipe_Project
│
├── data
│   └── keypoints.csv          # extracted landmark dataset
│
├── models
│   ├── asl_model.h5           # trained neural network
│   └── label_encoder.pkl      # encoder for class labels
│
├── scripts
│   ├── extract_keypoints.py   # dataset preprocessing
│   ├── test_model.py          # for testing model with camera
│   └── train_model.py         # neural network training
│
└── requirements.txt
```

---

# Dataset

The project uses the **ASL Alphabet Dataset** from Kaggle.
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Dataset structure looks like this:

```
asl_alphabet_train/
│
├── A
│   ├── A1.jpg
│   ├── A2.jpg
│
├── B
│   ├── B1.jpg
...
```

Each folder represents one ASL class.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ASL_MediaPipe_Project.git
cd ASL_MediaPipe_Project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Required libraries include:

* TensorFlow
* MediaPipe
* OpenCV
* NumPy
* Pandas
* Scikit-learn

---

# Step 1 — Extract Hand Keypoints

Script:

```
scripts/extract_keypoints.py
```

This script:

1. Loads images from the dataset
2. Detects hands using **MediaPipe**
3. Extracts **21 hand landmarks**
4. Converts them into **63 coordinates (x, y, z)**
5. Normalizes coordinates relative to the wrist
6. Saves all samples into a CSV dataset

Run:

```bash
python scripts/extract_keypoints.py
```

Output:

```
data/keypoints.csv
```

Each row contains:

```
label,x1,y1,z1,x2,y2,z2,...,x21,y21,z21
```

---

# Step 2 — Train the Neural Network

Script:

```
scripts/train_model.py
```

This script performs:

1. Loading the keypoint dataset
2. Normalizing coordinates
3. Encoding class labels
4. Splitting data into train/test sets
5. Training a neural network classifier
6. Saving the trained model

Run:

```bash
python scripts/train_model.py
```

Output:

```
models/asl_model.h5
models/label_encoder.pkl
```

The model uses a **Multilayer Perceptron (MLP)** architecture with:

* Dense layers
* Batch normalization
* Dropout regularization
* Early stopping

---

# Model Input Format

Each hand is represented as:

```
21 landmarks × 3 coordinates
```

Shape used by the neural network:

```
(21, 3)
```

Flattened into **63 features** during training.

---

# Training Features

The training pipeline includes:

* coordinate normalization
* wrist-relative positioning
* label encoding
* early stopping
* model checkpointing

These techniques improve generalization and training stability.

---

# Output Example

After training:

```
Test accuracy: 0.97
```
(Accuracy depends on your own dataset quality and landmark detection!)


---

# Future Improvements

Possible improvements include:

* real-time gesture recognition from webcam
* adding temporal models (LSTM)
* training on dynamic gestures
* improving dataset augmentation

---

# Author

Antony, student of Kiev Mohyla Academy
