# 🐶 Dog Breed Detection using InceptionV3

This project implements a deep learning model for **dog breed classification** using the **InceptionV3** architecture. The dataset consists of **70 dog breeds**, and the model was trained with **early stopping over 80 epochs**, achieving a high validation accuracy.

---

## 📈 Model Performance

- **Training Accuracy**: 96.50%
- **Validation Accuracy**: 91.67%
- **Validation Loss**: 1.9452
- **Epochs**: 80 (early stopping applied)

---

## 🧠 Model Architecture

- **Base Model**: InceptionV3 (pretrained on ImageNet)
- **Custom Top Layers**:
  - GlobalAveragePooling2D
  - Dense (256 units, ReLU)
  - Dropout (0.5)
  - Dense (70 units, Softmax)

---

## 🗂 Dataset

- **Total Classes**: 70 Dog Breeds
- **Input Shape**: Typically resized to 299x299 (as required by InceptionV3)
- **Preprocessing**: 
  - Rescaling
  - Data Augmentation (optional)
  - Label Encoding

---

## 🧪 Training Configuration

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Early Stopping**: Monitored `val_loss`, patience of N (you can specify)
- **Batch Size**: You can include this if known (e.g., 32 or 64)

---

## 📦 Requirements

```bash
tensorflow>=2.x
numpy
matplotlib
scikit-learn
pandas
