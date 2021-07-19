import numpy as np


def LU_factor(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i, i] = 1
        U[i, i] = (A[i, i] - (L[i, :i] @ U[:i, i])) / L[i, i]
        for j in range(i + 1, n):
            U[i, j] = (A[i, j] - (L[i, :i] @ U[:i, j])) / L[i, i]
        for k in range(i + 1, n):
            L[k, i] = (A[k, i] - (L[k, :i] @ U[:i, i])) / U[i, i]
    return L, U


def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double)

    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        y[i] = (b[i] - (L[i, :i] @ y[:i])) / L[i, i]

    return y


def back_substitution(U, y):
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double)

    x[-1] = y[-1] / U[-1, -1]

    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - (U[i, i:] @ x[i:])) / U[i, i]

    return x


n, m = map(int, input().split())
A = np.zeros((n, n))
result = []
for i in range(n):
    A[i] = np.array(list(map(int, input().split())))
for i in range(m):
    b = np.array(list(map(int, input().split()))).T
    L, U = LU_factor(A)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    x = np.round(x, 2)
    result.append(' '.join(map(str, x.T)))
print('\n'.join(map(str, result)))





