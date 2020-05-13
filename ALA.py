'''
ALA (Applied Linear Algebra)
----------------------------

Supplement Code to Linear Algebra Textbook:
Applied Linear Algebra
by Peter J. Olver and Chehrzad Shakiban

Methods:
    1. singular -> check if a matrix is singular
    2. plu -> permuted LU factorization
    3. inv -> inverse of a square matrix
    4. det -> determinant of a matrix
    5. coimg -> coimage of a matrix
    6. img -> image of a matrix
    7. ker -> kernel of a matrix
    8. coker -> cokernel of a matrix
    9. fss -> four fundemental subspaces of a matrix
    10. qr -> QR decomposition on a square matrix
    11. solvex -> solve Ax=b'''
import numpy as np 

def _row_swap(A, i, j):
    '''Swap two rows of a numpy ndarray.'''
    temp = A[j].copy()
    A[j] = A[i].copy()
    A[i] = temp
    return A

def singular(A):
    '''Check if a matrix is singular.'''
    cols = A.shape[0]
    U = np.array(A, dtype=float)
    shift = 0
    for i in range(1, cols): # RREF
        for j in range(0, i+shift):
            try:
                if U[i, j] != 0:
                    if U[j, j] == 0:
                        U = _row_swap(U, i, j)
                    else:
                        U[i] = U[i] - (U[i, j]/U[j, j])*U[j]
            except:
                break
        for j in range(i, A.shape[1]):
            if U[i, j] != 0:
                break
            shift += 1
    singular = False
    for i in range(0, cols): # Check for a pivot in each column
        try: 
            if U[i, i] == 0:
                singular = True
                break
        except:
            singular = True
            break
    return singular

def plu(A):
    '''Permuted LU factorization.'''
    L = np.identity(A.shape[0])
    U = np.array(A, dtype=float)
    P = np.identity(A.shape[0])
    shift = 0
    for i in range(1, A.shape[0]): # RREF
        for j in range(0, i+shift):
            try:
                if U[i, j] != 0:
                    if U[j, j] == 0:
                        U = _row_swap(U, i, j)
                        P = _row_swap(P, i, j)
                    else:
                        L[i, j] = U[i, j]/U[j, j]
                        U[i] = U[i] - (U[i, j]/U[j, j])*U[j]
            except:
                break
        for j in range(i, A.shape[1]):
            if U[i, j] != 0:
                break
            shift += 1  
    return P, L, U

def _fsub(L, b):
    '''Forward substitution on a lower triangular matrix.'''
    c = np.zeros(b.size)
    for i in range(c.size):
        if L[i, i] != 0:
            c[i] = (b[i] - np.sum(L[i, 0:i] * c[0:i])) / L[i, i]
        else:
            return None
    return c

def _bsub(U, c):
    '''Back substitution on an upper triangular matrix.'''
    x = np.zeros(c.size)
    for i in range(x.size):
        curr = x.size - i - 1
        if U[curr, curr] != 0:
            x[curr] = (c[curr] - np.sum(U[curr, curr+1:] * x[curr+1:])) / U[curr, curr]
        else:
            return None
    return x

def inv(A):
    '''Inverse of a square matrix.'''
    if A.shape[0] != A.shape[1]:
        raise AttributeError('Matrix A is non-square')
    if singular(A):
        raise AttributeError('Matrix A is Singular')
    I = np.identity(A.shape[0])
    aug = np.hstack([A, I]) # Build augmented matrix with I
    for i in range(0, aug.shape[0]): # Convert A into U
        for j in range(0, i):
            if aug[i, j] != 0:
                aug[i] = aug[i] - (aug[i, j]/aug[j, j])*aug[j]
    for i in range(A.shape[0] - 1, -1, -1): # Convert U into I
        aug[i] = aug[i] / aug[i, i]
        for j in range(0, i):
            aug[j] = aug[j] - aug[i] * aug[j, i]
    return aug[:, int(aug.shape[1]/2):]

def det(A):
    '''Determinant of a matrix.'''
    _, _, U = plu(A)
    d = 1
    for i in range(A.shape[0]):
        d = d * U[i, i]
    return d

def coimg(A):
    '''Comimage of a matrix'''
    _, _, U = plu(A)
    basis = []
    for row in range(U.shape[0]):
        if any(U[row, :]):
            vector = U[row, :]
            vector.shape = (vector.size, 1)
            basis.append(vector)
    return np.hstack(basis)

def img(A):
    '''Image of a matrix.'''
    return coimg(np.transpose(A))

def ker(A):
    '''Kernel of a matrix.'''
    _, _, U = plu(A)
    basic_variables = np.array([])
    for row in range(A.shape[0]): # Find indicied of basic variables
        for col in range(A.shape[1]):
            if U[row, col] != 0:
                basic_variables = np.append(basic_variables, col)
                break
    U = U[[row for row in range(A.shape[0]) if any(U[row])]]
    num_free_variables = A.shape[1] - basic_variables.size
    basis = np.zeros((A.shape[1], num_free_variables))
    free_counter = num_free_variables - 1
    for i in reversed(range(0, A.shape[1])): # Free variables have standard basis rows
        if np.isin(i, basic_variables) == 0:
            standard = np.zeros(num_free_variables)
            standard[free_counter] = 1
            free_counter -= 1
            basis[i] = standard
    for row, pivot in zip(reversed(U), reversed(basic_variables)): # Basic variables are linear combinations of lower variables
        srow = row/row[int(pivot)]
        for i in range(int(pivot)+1, A.shape[1]):
            basis[int(pivot)] += (-1)*srow[i]*basis[i]
    return basis

def coker(A):
    '''Cokernel of a matrix.'''
    return ker(np.transpose(A))

def ffs(A):
    '''Four fundemental subspaces of a matrix.
Return order:
    1. Kernel
    2. Image
    3. Coimge
    4. Cokernel'''
    return ker(A), img(A), coimg(A), coker(A)

def _mag(v):
    return np.sqrt(np.sum(v**2))

def qr(A):
    '''QR decomposition on a square matrix.'''
    if A.shape[0] != A.shape[1]:
        raise AttributeError('Non-square Matrix')
    Q = np.array(A, float)
    R = np.zeros((A.shape[0], A.shape[1]))
    for i in range(Q.shape[1]):
        mag = _mag(Q[:,i])
        Q[:,i] = Q[:,i] / mag # Normalize columns in Q
        R[i,i] = mag
        for j in range(i+1, A.shape[1]): # Orthogonalize columns in Q
            R[i,j] = np.dot(Q[:,i], Q[:,j])
            Q[:,j] = Q[:,j] - R[i,j] * Q[:,i]
    return Q, R

def solvex(A, b, householder=False):
    '''Solve Ax=b.'''
    if singular(A):
        raise AttributeError('Marix A is Singular')
    if householder:
        Q, R = qr(A)
        QTb = np.matmul(np.transpose(Q), b)
        x = _bsub(R, QTb)
        return x
    P, L, U = plu(A)
    b_hat = np.matmul(P, b)
    c = _fsub(L, b_hat)
    x = _bsub(U, c)  
    return x

def _pprint(t, ls, ms):
    if t:
        print(t)
    for l, m in zip(ls, ms):
        print(l)
        print(m)
    print()

if __name__ == '__main__':
    '''1. singular -> check if a matrix is singular
    2. plu -> permuted LU factorization
    3. inv -> inverse of a square matrix
    4. det -> determinant of a matrix
    5. coimg -> coimage of a matrix
    6. img -> image of a matrix
    7. ker -> kernel of a matrix
    8. coker -> cokernel of a matrix
    9. ffs -> four fundemental subspaces of a matrix
    10. qr -> QR decomposition on a square matrix
    11. solvex -> solve Ax=b'''
    A = np.array([[1, 1], [1, .1]])
    b = np.array([[1],[1]])
    ls = ['Matrix A:', 'Vector b']
    ms = [A, b]
    _pprint(None, ls, ms)
    _pprint(None, ['Is singular:'], [singular(A)])
    _pprint('Permuted LU factorization of A:', ['P:', 'L:', 'U:'], plu(A))
    _pprint(None, ['Inverse of A:'], [inv(A)])
    _pprint(None, ['Determinant of A:'], [det(A)])
    _pprint('Four fundemental subspaces of A:', ['Kernel:', 'Image:', 'Coimage:', 'Cokernel:'], ffs(A))
    _pprint('QR decomposition:', ['Q:', 'R:'], qr(A))
    _pprint('Solve Ax=b (PLU):', ['x:'], [solvex(A, b)])
    _pprint('Solve Ax=b (QR):', ['x:'], [solvex(A, b, True)])