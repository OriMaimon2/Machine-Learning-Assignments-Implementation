# Machine-Learning-Assignments-Implementation

This repository contains solutions to multiple assignments from a Statistical / Machine Learning course. It includes both theoretical derivations and practical implementations of key algorithms, along with experiments and visualizations.

The repository covers the following topics:
- Linear Regression
- Logistic Regression
- K-Means Clustering
- Eigenfaces for Face Recognition (PCA-based)

All implementations are provided along with a submission PDF containing results and explanations.

---

## 📂 Repository Structure
├── linear_regression/
├── logistic_regression/
├── kmeans/
├── eigenfaces/
├── report.pdf
└── README.md


---

## 📘 Assignments Overview

### 1. Linear Regression
Based on the standard linear model:

\[
Y = X\beta + \epsilon
\]

The objective is to estimate the parameter vector:

\[
\hat{\beta} = (X^T X)^{-1} X^T Y
\]

#### Tasks:
- Generate synthetic dataset \(X\), true parameters \(\beta\), and Gaussian noise.
- Compute target values \(Y\).
- Estimate parameters using the closed-form solution.
- Compare estimated parameters with ground truth.
- Analyze the effect of increasing noise variance on estimation accuracy.

📎 See assignment details: :contentReference[oaicite:0]{index=0}

---

### 2. Logistic Regression
Binary classification using the sigmoid function:

\[
g(z) = \frac{1}{1 + e^{-z}}
\]

#### Tasks:
- Derive gradient of the log-likelihood function.
- Implement Logistic Regression using Gradient Ascent.
- Train a classifier to distinguish between digits (e.g., 1 vs 2).
- Track and plot the log-likelihood during training.
- Evaluate classification accuracy.

📎 See assignment details: :contentReference[oaicite:1]{index=1}

---

### 3. K-Means Clustering
Unsupervised clustering algorithm minimizing:

\[
J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - c_i\|^2
\]

#### Tasks:
- Implement K-Means from scratch.
- Apply the algorithm on the MNIST dataset with \(K=10\).
- Track convergence and verify cost decreases each iteration.
- Evaluate clustering performance (label matching).

📎 See assignment details: :contentReference[oaicite:2]{index=2}

---

### 4. Eigenfaces Face Recognition (PCA)
Nearest-neighbor face recognition using a low-dimensional PCA representation.

#### Dataset:
- 15 individuals
- 11 grayscale images per person
- Image size: 32×32 (1024-dimensional space)

#### Workflow:
1. Split dataset:
   - 8 images per person → training
   - 3 images per person → testing

2. Compute:
   - Mean face
   - PCA (Eigenfaces)

3. Dimensionality Reduction:
   - Extract top \(K\) principal components
   - Compare custom PCA implementation with MATLAB `princomp`

4. Projection:
   - Project both training and test images into the eigenface space

5. Classification:
   - Use nearest neighbor (Euclidean distance) in reduced space

#### Outputs:
- Mean face visualization
- Top 20 eigenfaces
- Classification accuracy vs. \(K\) plot

#### Expected Behavior:
- Low accuracy for very small \(K\)
- Rapid improvement as \(K\) increases
- Performance saturation at higher dimensions

---

## ⚙️ Technologies Used

- Python / MATLAB (depending on implementation)
- NumPy / SciPy
- Matplotlib (visualizations)
- scikit-learn (for comparison where relevant)

---

## 📊 Results

The repository includes:
- Visualizations (plots, eigenfaces, convergence curves)
- Accuracy metrics for classification tasks
- Analysis of model behavior under different conditions

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

Run each module independently:
python linear_regression/main.py
python logistic_regression/train.py
python kmeans/kmeans.py
python eigenfaces/eigenfaces.py


📎 Notes
All implementations are written from scratch unless otherwise specified.
Some parts include comparisons to built-in functions (e.g., PCA).
The repository is intended for educational and research purposes.

Ori Maimon
