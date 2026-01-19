# Logistic Regression From Scratch (NumPy)

A small, from-scratch logistic regression exercise using NumPy. The notebook covers a basic 2D dataset and a second dataset that uses feature mapping with regularized logistic regression, then visualizes the data and decision boundaries.

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- Pandas (imported in the notebook)

Install dependencies (example):

```bash
pip install numpy matplotlib pandas
```

## How to run
1. Start Jupyter (or open the notebook in VS Code).
2. Run the notebook in order.

```bash
jupyter notebook logistic_regression_using_numpy.ipynb
```

## Project layout
- `logistic_regression_using_numpy.ipynb`: main notebook with model, training, and plots.
- `utils.py`: helper for loading the dataset.
- `data/ex2data1.txt`: 2-feature dataset with a binary label in the last column.
- `data/ex2data2.txt`: dataset used for feature mapping and regularized logistic regression.

## Notes
- The decision boundary for a 2-feature logistic model is a straight line defined by `w0*x1 + w1*x2 + b = 0`.
- For the mapped-feature dataset, the boundary is non-linear due to polynomial feature expansion and regularization.
- The notebook assumes the dataset is already clean and numeric.
