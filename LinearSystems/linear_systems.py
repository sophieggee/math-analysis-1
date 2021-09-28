# linear_systems.py
"""Volume 1: Linear Systems.
<Sophie Gee>
<section 3>
<9.28>
"""
import numpy as np
from scipy import linalg as la
import time
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as spla

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    n = A.shape[0]

    #row reduction
    for j in range(n-1):            #iterates through rows until last row
        for i in range(j+1, n):   #iterates through rows after i
            c = A[i,j]/A[j,j]       #finds scalar for multiplication
            A[i,j+1:n] = A[i,j+1:n] - c*A[j,j+1:n]  #subtract the scalar multiple of the upper row from the lower row
            A[i,j] = 0
    return A


# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    n = A.shape[0] #returns dimension n of square matrix A
    U = A.copy() #creates copy of A
    L = np.identity(n) #identity of size n 
    for j in range(n): # for j= 0,..,n-1
        for i in range(j+1, n): # for i= j+1,...,n-1
            L[i,j] = U[i,j]/U[j,j] #divides rows in U for lower triangular matrix
            U[i,j:] = U[i,j:] - L[i,j]*U[j,j:] #sets U to be row reduced
    return L,U

# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    n = A.shape[0]
    L,U = lu(A)
    y = np.zeros((n))
    x = np.zeros((n))
    sum = 0

    for k in range(n): #finds sum for each inidvidual y_k
        for j in range(k):
            sum += L[k,j]*y[j]
        y[k] = b[k] - sum #returns new y

    sum2=0    
    for k in range(n-1, -1, -1): #finds sum for each individual x_k using y_k
        for j in range(k, n):
            sum2 += U[k,j]*x[j]
        x[k] = 1/U[k,k]*((y[k]) - sum2) #returns new x
    
    return x



# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    
    domain = 2**np.arange(1,13)
    times =[]
    times2=[]
    times3=[]
    times4=[]
    for n in domain:
        A = np.random.random((n,n))
        b = np.random.random(n)
        start = time.time()
        np.matmul(la.inv(A),b)
        times.append(time.time() - start)

        start2 = time.time()
        la.solve(A,b)
        times2.append(time.time() - start2)

        start3 = time.time()
        L, P = la.lu_factor(A)
        la.lu_solve((L,P), b)
        times3.append(time.time() - start3)

        L, P = la.lu_factor(A)
        start4 = time.time()
        la.lu_solve((L,P), b)
        times4.append(time.time() - start4)

    plt.plot(domain, times, '.-', linewidth=2, markersize=15, label="invert then left-multiply")
    plt.plot(domain, times2, '.-', linewidth=2, markersize=15, label="using linalg.solve")
    plt.plot(domain, times3, '.-', linewidth=2, markersize=15, label="using linalg.lu_solve and linalg.lu_factor")
    plt.plot(domain, times4, '.-', linewidth=2, markersize=15, label="using linalg.lu_solve")
    plt.legend()
    plt.ylabel("time taken to compute")
    plt.xlabel("matrices of size nxn")
    plt.show()


# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    B = sparse.diags([1,-4,1],[-1,0,1],shape=(n,n)) #create B
    A = sparse.block_diag([B]*n) #set A diagonals to B matrices
    A.setdiag(np.ones(n**2), n) #set other diagonals to identity matrices of size n
    A.setdiag(np.ones(n**2), -n)
    return A


# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    domain = 2**np.arange(1,8)
    times =[]
    times2=[]
    for n in domain:
        A = prob5(n)
        b = np.random.random(n**2)
        Acsr = A.tocsr()
        start = time.time()
        spla.spsolve(Acsr, b)
        times.append(time.time() - start)

        A = Acsr.toarray()
        start2 = time.time()
        la.solve(A,b)
        times2.append(time.time() - start2)

    plt.plot(domain, times, '.-', linewidth=2, markersize=15, label="using csr spsolve")
    plt.plot(domain, times2, '.-', linewidth=2, markersize=15, label="using linalg.solve")
    plt.ylabel("time taken to compute")
    plt.xlabel("matrices of size n**2")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    prob6()