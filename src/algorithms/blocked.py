import numpy as np

def blocked_dot(A, B, block=64):
    """Blocked matrix multiplication using numpy.dot inside each block."""
    n = A.shape[0]
    C = np.zeros((n, n))

    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        for j0 in range(0, n, block):
            j1 = min(j0 + block, n)
            for k0 in range(0, n, block):
                k1 = min(k0 + block, n)
                C[i0:i1, j0:j1] += A[i0:i1, k0:k1].dot(B[k0:k1, j0:j1])

    return C
