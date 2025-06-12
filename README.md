# Linear-and-Logistic-Regression-with-Gradient-Descent

Machine Learning Fundamentals: Linear and Logistic Regression from Scratch
This repository contains implementations of two fundamental machine learning algorithms: Linear Regression and Logistic Regression. The core focus of these implementations is to demonstrate how these models can be built from first principles using Gradient Descent for parameter optimization, rather than relying on pre-built library functions for the core fitting process.

This project was developed through two Jupyter notebooks:

lin_reg_solver.ipynb

log_reg_solver.ipynb

Implemented Concepts
1. Linear Regression
Model: Standard linear model  
y
^
​
 =w 
T
 x+w 
0
​
 .

Loss Function: Mean Squared Error (MSE) / Sum of Squares Loss: L= 
N
1
​
 ∑ 
i=1
N
​
 (y 
i
​
 − 
y
^
​
 ) 
2
 .

Optimization: Batch Gradient Descent, with explicit calculation of the gradient ∇L=−2⋅ 
N
1
​
 ⋅X 
T
 (y− 
y
^
​
 ) and iterative weight updates.

Convergence: Implemented a stopping criterion based on the change in the gradient's norm (ϵ).

2. Logistic Regression
Classification Task: Designed for binary classification problems.

Data Generation: Use of sklearn.datasets.make_classification to create synthetic datasets with controlled n_features, n_redundant, n_informative, and n_repeated parameters to analyze feature impact.

Feature Scaling: Essential standardization (z-score normalization) of features:

x 
i
′
 j
​
 = 
σ 
j
​
 
x 
i
j
​
 −μ 
j
​
 
​
 
applied using training set statistics.

Activation Function: Sigmoid (logistic) function: σ(z)= 
1+e 
−z
 
1
​
 .

Loss Function: Cross-Entropy Loss, implemented for efficient matrix operations:

L(w)=− 
i=1
∑
N
​
 [y 
i
​
 log(σ(X 
i
​
 w))+(1−y 
i
​
 )log(1−σ(X 
i
​
 w))]
Optimization: Batch Gradient Descent, with iterative weight updates:

w 
(τ+1)
 =w 
(τ)
 + 
N
α
​
 X 
T
 (y−σ(Xw))
Prediction: Binary classification based on the sign of Xw.

Workflow: Standard machine learning workflow including data splitting (training/testing) and loss visualization to confirm convergence.

Project Structure
lin_reg_solver.ipynb: Jupyter notebook containing the Linear Regression implementation.

log_reg_solver.ipynb: Jupyter notebook containing the Logistic Regression implementation.

How to Run
To run these notebooks, you will need:

Python 3.x

Jupyter Notebook or JupyterLab

Common scientific libraries: numpy, pandas, matplotlib, sklearn

Simply open the .ipynb files in Jupyter and execute the cells.

Conclusion
This project serves as a practical exploration of the core mechanisms behind linear and logistic regression models, emphasizing the iterative optimization process through gradient descent. It provides a foundational understanding of how these algorithms learn from data and the importance of preprocessing steps like feature scaling.
