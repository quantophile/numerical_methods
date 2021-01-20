import numpy as np
import numpy.linalg as linalg


def polyapp(x, y, n):
    m = len(x)  # No of data points
    a = min(x)  # Determine the interval [a,b]
    b = max(x)
    d = (a + b) / 2

    # We assume a shifted power basis for a polynomial p in P_n.
    def p(x, k):
        return (x - d) ** k

    # We find the coefficient vector c for a polynomial p in P_n, such that
    # y_i = p(x_i) in a least squares sense. The coefficient vector is given by the solution of a square linear
    # algebraic system.

    A = np.zeros((n, n))

    # The term multiplying c_j in the k'th equation is sum_{i=1}^m p_k(x_i) p_j(x_i)

    for k in range(0, n):
        for j in range(0, n):
            for i in range(0, m):
                A[k, j] = A[k, j] + p(x[i], k) * p(x[i], j)

    # Construct the right hand side vector
    b = np.zeros(m)

    for k in range(0, n):
        for i in range(0, m):
            b[k] = b[k] + y[i] * p(x[i], k)

    c = linalg.solve(A, b)
    return A, c


def polyval(x, c, d, n):
    y = 0
    for j in range(0, n):
        y = y + c[j] * ((x - d) ** j)
    return y