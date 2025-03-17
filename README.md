Mathematical number detection Using CNN & Traditional Machine Learning

Project Overview

This project focuses on classifying mathematical symbols (e.g., numbers, operators) using a Convolutional Neural Network (CNN) and traditional machine learning models. The dataset is preprocessed and trained using TensorFlow/Keras for deep learning and Scikit-Learn for machine learning models like k-NN, SVM, Decision Tree, Random Forest, and AdaBoost.

Features

Deep Learning Model: CNN architecture trained on image datasets.

Machine Learning Models: k-NN, SVM, Decision Tree, Random Forest, and AdaBoost classifiers.

Evaluation Metrics: Accuracy, Confusion Matrix.

Data Augmentation: To improve generalization in CNN.

Visualization: Matplotlib for training curves and confusion matrices.

Dataset

The dataset consists of images of symbols such as numbers and mathematical operators. It is split into training and validation sets.

Dataset Preprocessing

Images are resized to a standard dimension.

Normalization applied to scale pixel values between 0 and 1.

Data augmentation applied for improving model generalization.

Model Implementation

1. CNN Model

Architecture:

Convolutional Layers with ReLU activation

MaxPooling for feature reduction

Dropout to reduce overfitting

Fully connected layers with Softmax activation

Training: Adam optimizer with categorical cross-entropy loss function.

2. Machine Learning Models

k-Nearest Neighbors (k-NN)

Support Vector Machine (SVM)

Decision Tree

Random Forest

AdaBoost

Training and Evaluation

CNN is trained using TensorFlow/Keras with data augmentation.

Machine learning models are trained using flattened image vectors.

Accuracy and confusion matrix are used for evaluation.

Results

CNN achieved ~88.9% validation accuracy, with signs of overfitting.

Machine learning models showed varying accuracy, with Random Forest performing best among them.

Challenges & Solutions

1. Overfitting in CNN

Implemented Dropout layers.

Increased data augmentation techniques.

2. Shape Mismatch in ML Models

Flattened images before feeding into ML models.

3. Evaluation on Full Dataset

Ensured full validation set was used for predictions.

Installation & Setup

Requirements

Python 3.8+

TensorFlow/Keras

NumPy, Pandas

Matplotlib, Seaborn

Scikit-Learn

Steps to Run

Clone the repository:

git clone https://github.com/your-repo/symbol-classification.git
cd symbol-classification

Install dependencies:

pip install -r requirements.txt

Run the training script:

python train.py

Future Improvements

Implement transfer learning (e.g., MobileNet, ResNet).

Tune hyperparameters for better performance.

Extend the dataset with additional symbols.

