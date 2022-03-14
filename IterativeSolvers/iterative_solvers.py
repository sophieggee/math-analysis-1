# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Sophie Gee>
<3/7/22>
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy import sparse as sp

# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot= False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    #set n to be the length of A, get intial x's
    n = len(A)
    x0, x1 = np.zeros(n), np.zeros(n)
    errs = []

    #iterate through maxiter times
    for i in range(maxiter):
        x1 = x0 + ((b - A.dot(x0)) / np.diag(A))
        err = la.norm(x1 - x0, ord=np.inf)
        errs.append(err)
        #if the error is less than tolerance, break
        if err < tol:
            break
        x0 = x1
        
    #plot for probklem 2
    if plot:
        plt.semilogy(errs)
        plt.xlabel("Iteration")
        plt.ylabel("Abs Error of Approximation")
        plt.title("Convergence of Jacobi Method")
        plt.show()
    #return solution to Ax = b
    return x1


# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    #get length of A, create zeros for x, collect errors
    n = len(A)
    x = np.zeros(n)
    errs = []

    #iterate through maxiter times
    for i in range(maxiter):
        prev = x.copy()
        for j in range(n):
            x[j] = x[j] + ((b[j] - np.dot(A[j].T, x)) / A[j,j]) 
        err = la.norm(x - prev, ord=np.inf)
        errs.append(err)
        #check error and tolerance
        if err < tol:
            break
    #check for plot boolean
    if plot:
        plt.semilogy(errs)
        plt.xlabel("Iteration")
        plt.ylabel("Abs Error of Approximation")
        plt.title("Convergence of Gauss Seidel Method")
        plt.show()

    #output solution to Ax = b
    return x


# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    #get n the shape of A, set initial variables
    n = A.shape[0]
    x = np.zeros(n)
    errs = []

    #range through maxiter times
    for i in range(maxiter):
        prev = x.copy()
        for j in range(n):
            start = A.indptr[j]
            end = A.indptr[j+1]
            Aix = A.data[start:end] @ x[A.indices[start:end]]
            x[j] = x[j] + ((b[j] - Aix) / A[j,j])
        err = la.norm(x - prev, ord=np.inf)
        #include this error norm
        errs.append(err)
        if err < tol:
            break

    #return optimal x
    return x


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    #set up initial variables, collect right dimensions
    n = A.shape[0]
    x = np.zeros(n)
    errs = []
    conv = False

    #iterate through until convergent, check convergence, output returns
    for i in range(maxiter):
        prev = x.copy()
        for j in range(n):
            start = A.indptr[j]
            end = A.indptr[j+1]
            Aix = A.data[start:end] @ x[A.indices[start:end]]
            x[j] = x[j] + ((b[j] - Aix) * (omega / A[j,j]))
        err = la.norm(x - prev, ord=np.inf)
        errs.append(err)
        if err < tol:
            conv = True
            break

    return x, conv, i + 1

# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    # constuct A, B, n squared
    ns = n ** 2
    B = sp.diags([1, -4, 1], [-1, 0, 1], shape=(n, n))
    A = sp.block_diag([B] * n)
    A += sp.diags([1, 1], [-n, n], shape=(ns, ns))
    
    # Contruct b
    base = np.zeros(n)
    base[0], base[-1] = -100, -100
    b = np.tile(base, ns)
    
    #collect solution to Ax = b
    sol = sor(A, b, omega, tol, maxiter)

    #check plot boolean
    if plot:
        u = sol[0].reshape((n, n))
        plt.pcolormesh(u, cmap="coolwarm")
        plt.title("Solution")
        plt.show()
    
    #return solution
    return sol


# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    #set up domain and interations of hot plate
    domain = np.arange(1, 2, .05)
    iters = [hot_plate(20, omega, tol=1e-2, maxiter=1000)[2] 
             for omega in domain]

    #plot the running of hot_plate
    plt.plot(domain, iters)
    plt.xlabel("Omega")
    plt.ylabel("Iteration")
    plt.title("Hot Plate")
    plt.show()
