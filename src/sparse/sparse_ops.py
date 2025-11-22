import scipy.sparse as sp

def scipy_sparse_mult(A, B):
    """Sparse matrix multiplication using CSR format."""
    A_sp = sp.csr_matrix(A)
    B_sp = sp.csr_matrix(B)
    return A_sp.dot(B_sp)
