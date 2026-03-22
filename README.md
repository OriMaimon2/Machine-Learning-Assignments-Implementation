# Machine-Learning-Assignments-Implementation

Course assignments in machine learning, implemented in both Python and MATLAB.

This README describes the repository as it currently exists in this folder.

## Repository Layout

```text
Machine-Learning-Assignments-Implementation/
|-- README.md
|-- linear_regression/
|   |-- HW1_Machine Learning.py
|   `-- linear_regression.pdf
|-- logistic_regression/
|   |-- HW_LM_2.m
|   |-- HW_LM_2.asv
|   |-- mnist_all.zip
|   |-- logit-hw.pdf
|   `-- תרגיל בית 2.pdf
|-- kmeans/
|   |-- HW_LM_3.m
|   `-- k-means.pdf
`-- eigenfaces/
    |-- HW_LM_5.m
    |-- HW_LM_5.asv
    |-- facesData.zip
    |-- pca.pdf
    |-- meanFace.png
    |-- eigFace.png
    `-- RateAsFuncOfK.png
```

## Folder-by-Folder Description

### `linear_regression/`
- Main script: `HW1_Machine Learning.py`
- Language: Python
- What it does:
- Creates a synthetic linear model `Y = Xb + e`
- Solves for `b` with the closed-form estimator `(X^T X)^-1 X^T Y`
- Prints estimation error and repeats with increasing Gaussian noise
- Notes:
- Uses `numpy` only
- No external dataset required

### `logistic_regression/`
- Main script: `HW_LM_2.m`
- Language: MATLAB
- What it does:
- Loads MNIST data from `mnist_all.mat`
- Trains binary logistic regression (digits 1 vs 2) using gradient ascent
- Plots log-likelihood over iterations
- Computes test success rate
- Files included:
- `mnist_all.zip` (dataset archive)
- Assignment PDFs: `logit-hw.pdf`, `תרגיל בית 2.pdf`

### `kmeans/`
- Main script: `HW_LM_3.m`
- Language: MATLAB
- What it does:
- Loads MNIST from `mnist_all.mat`
- Runs K-means with `K = 10` on digit images
- Plots clustering cost across iterations
- Computes a simple label-matching success rate
- Files included:
- Assignment PDF: `k-means.pdf`

### `eigenfaces/`
- Main script: `HW_LM_5.m`
- Language: MATLAB
- What it does:
- Loads face dataset from `facesData.mat`
- Splits images into train/test per subject
- Computes mean face and PCA eigenfaces
- Runs nearest-neighbor recognition in reduced PCA space
- Saves figures:
- `meanFace.png`
- `eigFace.png`
- `RateAsFuncOfK.png`
- Files included:
- `facesData.zip` (dataset archive)
- Assignment PDF: `pca.pdf`

## Requirements

- Python 3 + `numpy` for `linear_regression/HW1_Machine Learning.py`
- MATLAB for all `.m` scripts
- Extract dataset archives before running MATLAB scripts:
- `logistic_regression/mnist_all.zip` -> `mnist_all.mat`
- `eigenfaces/facesData.zip` -> `facesData.mat`

## How to Run

### Python assignment

```bash
cd linear_regression
python "HW1_Machine Learning.py"
```

### MATLAB assignments

Run each script from its own folder in MATLAB so relative `load(...)` calls work.

```matlab
% Logistic regression
cd logistic_regression
HW_LM_2

% K-means
cd ../kmeans
HW_LM_3

% Eigenfaces
cd ../eigenfaces
HW_LM_5
```

## Notes

- `.asv` files are MATLAB autosave/backups.
- Some assignment instructions are in Hebrew PDF files.
- Generated images in `eigenfaces/` are output artifacts from the script.



Ori Maimon
