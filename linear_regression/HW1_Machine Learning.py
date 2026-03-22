import numpy
from numpy import random
from numpy import linalg

if __name__ == '__main__':
    n = 7
    m = 3
    X = random.rand(n, m)  # P1
    b = [[1], [2], [3]]  # P2
    e = random.rand(n, 1)  # P3
    Y = (X @ b) + e  # P4
    b_solv = linalg.inv(X.T @ X) @ X.T @ Y  # P5
    error = linalg.norm(b_solv - b)
    print("\n", "The Euclidean distance between the original vector b and the vector b we calculated: ", error, "\n")  # P6
    for i in [1, 2, 3, 4, 5, 6, 7]:   # P7
        e_j = random.normal(0, pow(3, i), (n, 1))
        Y_j = (X @ b) + e_j
        b_solv_j = linalg.inv(X.T @ X) @ X.T @ Y_j
        error_j = linalg.norm(b_solv_j - b)
        print("For sigma = ", i, " the error is ", error_j, "\n")
    #We will notice that as sigma increases, the error increases.
    #This makes sense because as sigma increases, there is a greater chance that e will be large and as a result Y will be less consistent with the model.
