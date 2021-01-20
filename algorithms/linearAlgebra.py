# Helper Function
import numpy as np
import matplotlib.pyplot as plt


def gaussJ(A, b):
    # Linear equation solution by Gauss-Jordan elimination with partial pivoting.
    # The input matrix is A[0..n-1][0..n-1] and b[0..n-1][0..m-1] is the input
    # containing m right hand side vectors.

    n, _ = A.shape
    _, m = b.shape

    # This is the main loop over the columns to be reduced
    for k in range(0, n):

        # Partial pivoting
        idx = k
        if A[k, k] == 0.0:
            big = 0.0
            for j in range(k + 1, n):
                if (A[j, k] > big):
                    big = A[j, k]
                    idx = j

        # We now have the pivot element, so we interchange the rows, if needed
        # to put the pivot element on the diagonal.
        if idx != k:
            for l in range(0, n):
                temp = A[k, l]
                A[k, l] = A[idx, l]
                A[idx, l] = temp

            for l in range(0, m):
                temp = b[k, l]
                b[k, l] = b[idx, l]
                b[idx, l] = temp

        # We divide the pivot row by the pivot element so that, the pivot is unity
        pivinv = 1.0 / A[k, k]
        for l in range(0, n):
            A[k, l] = A[k, l] * pivinv
        for l in range(0, m):
            b[k, l] = b[k, l] * pivinv

        # Next we reduce the rows except the pivot one
        for i in range(k + 1, n):
            mult = A[i, k]
            for j in range(0, n):
                A[i, j] = A[i, j] - mult * A[k, j]

            for j in range(0, m):
                b[i, j] = b[i, j] - mult * b[k, j]

    return A, b