'''
MatMath
-------

Provides
    1. Solve linear systems of equations (Ax=b)
    2. Find the inverse of a matrix'''
import numpy as np 
import pandas as pd 

def _row_swap(A, i, j):
    '''Swap two rows of a numpy ndarray'''
    temp = A[j].copy()
    A[j] = A[i].copy()
    A[i] = temp
    return A

def _factor_permuted_lu(A):
    '''Factorize a matrix into its permuted LU form'''
    L = np.identity(A.shape[0])
    U = np.array(A, dtype=float)
    P = np.identity(A.shape[0])
    for i in range(1, A.shape[0]):
        for j in range(0, i):
            if U[i, j] != 0:
                if U[j, j] == 0:
                    U = _row_swap(U, i, j)
                    P = _row_swap(P, i, j)
                else:
                    L[i, j] = U[i, j]/U[j, j]
                    U[i] = U[i] - (U[i, j]/U[j, j])*U[j]
    for i in range(0, A.shape[0]):
        if U[i, i] == 0:
            return None, None, None
    return P, L, U

def _fsub(L, b):
    '''Forward substitution on a lower triangular matrix'''
    c = np.zeros(b.size)
    for i in range(c.size):
        if L[i, i] != 0:
            c[i] = (b[i] - np.sum(L[i, 0:i] * c[0:i])) / L[i, i]
        else:
            return None
    return c

def _bsub(U, c):
    '''Back substitution on an upper triangular matrix'''
    x = np.zeros(c.size)
    for i in range(x.size):
        curr = x.size - i - 1
        if U[curr, curr] != 0:
            x[curr] = (c[curr] - np.sum(U[curr, curr+1:] * x[curr+1:])) / U[curr, curr]
        else:
            return None
    return x

def solvex(A, b):
    '''solvex(A, b)
Permuted LU solution of Ax=b.

Parameters
----------
A : ndarray, required
    The n by n coefficient matrix.
b : ndarray, required
    The n by 1 solutions to the linear system.
    
Returns
-------
x : ndarray
    Array of the same shape as `b`, containing the solutions
    to the linear system.

Notes
-----
A needs to be nonsingular for a solution.

Examples
--------
>>> solvex(np.array([[0, 1, -3], [0, 2, 3], [1, 0, 2]]), np.array([1, 2, -1]))
[-1. 1. -0.]'''
    P, L, U = _factor_permuted_lu(A)
    b_hat = np.matmul(P, b)
    c = _fsub(L, b_hat)
    x = _bsub(U, c)  
    return x

def inverse(A):
    '''inverse(A)
Find the matrix inverse of A.

Parameters
----------
A : ndarray, required
    Nonsingular matrix.

Returns
-------
X : ndarray
    Matrix inverse of A.

Notes
-----
A needs to be nonsingular for a solution.

Examples
--------
>>> inverse(np.array([[1, -2], [3, -3]]))
[[-1.       0.66666667]
 [-1.       0.33333333]]'''
    I = np.identity(A.shape[0])
    aug = np.hstack([A, I])
    for i in range(0, aug.shape[0]):
        if aug[i, i] == 0:
            return None
        for j in range(0, i):
            if aug[i, j] != 0:
                aug[i] = aug[i] - (aug[i, j]/aug[j, j])*aug[j]
    for i in range(A.shape[0] - 1, -1, -1):
        aug[i] = aug[i] / aug[i, i]
        for j in range(0, i):
            aug[j] = aug[j] - aug[i] * aug[j, i]
    X = aug[:, int(aug.shape[1]/2):]
    return X