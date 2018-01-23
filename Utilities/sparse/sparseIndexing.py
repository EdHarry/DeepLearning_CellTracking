import scipy
import scipy.sparse

__author__ = 'edwardharry'


def sparse_max_row(csr_mat):
    if ~scipy.sparse.isspmatrix_csr(csr_mat):
        csr_mat = scipy.sparse.csr_matrix(csr_mat)

    ret = scipy.zeros(csr_mat.shape[0])
    ret[scipy.diff(csr_mat.indptr) != 0] = scipy.maximum.reduceat(csr_mat.data,
                                                                  csr_mat.indptr[:-1][scipy.diff(csr_mat.indptr) > 0])
    return ret


def sparse_min_row(csr_mat):
    if ~scipy.sparse.isspmatrix_csr(csr_mat):
        csr_mat = scipy.sparse.csr_matrix(csr_mat)

    ret = scipy.zeros(csr_mat.shape[0])
    ret[scipy.diff(csr_mat.indptr) != 0] = scipy.minimum.reduceat(csr_mat.data,
                                                                  csr_mat.indptr[:-1][scipy.diff(csr_mat.indptr) > 0])
    return ret


def sparse_sum_row(csr_mat):
    if ~scipy.sparse.isspmatrix_csr(csr_mat):
        csr_mat = scipy.sparse.csr_matrix(csr_mat)

    ret = scipy.zeros(csr_mat.shape[0])
    ret[scipy.diff(csr_mat.indptr) != 0] = scipy.add.reduceat(csr_mat.data,
                                                              csr_mat.indptr[:-1][scipy.diff(csr_mat.indptr) > 0])
    return ret


def sparse_max_col(csr_mat):
    if ~scipy.sparse.isspmatrix_csc(csr_mat):
        csr_mat = scipy.sparse.csc_matrix(csr_mat)

    ret = scipy.zeros(csr_mat.shape[1])
    ret[scipy.diff(csr_mat.indptr) != 0] = scipy.maximum.reduceat(csr_mat.data,
                                                                  csr_mat.indptr[:-1][scipy.diff(csr_mat.indptr) > 0])
    return ret


def sparse_min_col(csr_mat):
    if ~scipy.sparse.isspmatrix_csc(csr_mat):
        csr_mat = scipy.sparse.csc_matrix(csr_mat)

    ret = scipy.zeros(csr_mat.shape[1])
    ret[scipy.diff(csr_mat.indptr) != 0] = scipy.minimum.reduceat(csr_mat.data,
                                                                  csr_mat.indptr[:-1][scipy.diff(csr_mat.indptr) > 0])
    return ret


def sparse_sum_col(csr_mat):
    if ~scipy.sparse.isspmatrix_csc(csr_mat):
        csr_mat = scipy.sparse.csc_matrix(csr_mat)

    ret = scipy.zeros(csr_mat.shape[1])
    ret[scipy.diff(csr_mat.indptr) != 0] = scipy.add.reduceat(csr_mat.data,
                                                              csr_mat.indptr[:-1][scipy.diff(csr_mat.indptr) > 0])
    return ret

# from https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
def delete_row_csr(mat, i):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

def delete_row_lil(mat, i):
    if not isinstance(mat, scipy.sparse.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])