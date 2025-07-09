# 🧠 Breast Cancer Classification with Neural Networks

This project uses a deep learning model built with **Keras** to classify breast cancer tumors as **benign (2)** or **malignant (4)** based on the **Breast Cancer Wisconsin (Original)** dataset.

## 📊 Dataset

- Dataset: [Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)
- Format: CSV
- Missing values: Represented by `?`, replaced with the column mean using `SimpleImputer`.

## 🧪 Features

The dataset contains the following features:
- Clump Thickness
- Uniformity of Cell Size
- Uniformity of Cell Shape
- Marginal Adhesion
- Single Epithelial Cell Size
- Bare Nuclei
- Bland Chromatin
- Normal Nucleoli
- Mitoses

> 🔍 The target class is the **10th column** which represents:
> - `2` = benign  
> - `4` = malignant

## 🧠 Model Architecture
    
The model is a simple feedforward neural network:
- Input layer: 8 neurons
- Hidden layers:
  - Dense (256 units, ReLU)
  - Dense (256 units, ReLU)
  - Dense (128 units, Softmax)
- Output: 2-class classification (sparse categorical cross-entropy)

## ⚙️ Training

- Optimizer: Adam  
- Loss: sparse_categorical_crossentropy  
- Epochs: 50  
- Batch size: 32  
- Validation split: 13%

## 🧾 Prediction Example

The model can predict the class of new tumor data. Example:

sample = np.array([10,5,5,3,6,7,7,10]).reshape(1, 8)
prediction = model.predict(sample)
Output is then mapped to:

0 → 2 (Benign)
1 → 4 (Malignant)


🚀 Getting Started
1. Clone this repository:
   ```bash
   git clone https://github.com/erenntorun/breast-cancer-classification-with-neural-networks.git
   
2. Navigate to the project folder:
   ```bash
   cd breast-cancer-classification-with-neural-networks

3. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn keras

4. Make sure your dataset path is correct in the script.

5. Run the script:
   ```bash
   python KanserTespiti.py


📌 Notes
The dataset contains missing values, replaced with the column mean.

The id column is dropped and not used in training.

Model is trained using only 2 classes.

Final prediction is post-processed to return class label as 2 or 4.




Created by @eren 👨‍💻
