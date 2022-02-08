# condition_stability.py
"""Volume 1: Conditioning and Stability.
<Sophie Gee>
<2/1/22>
"""

import numpy as np
import sympy as sy
import scipy.linalg as la
from matplotlib import pyplot as plt


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    sig1 = max(la.svdvals(A))
    sign = min(la.svdvals(A))

    #check if smallest value is 0, then return inf
    if sign == 0:
        return np.inf
    #otherwise return equation 5.3
    else:
        return sig1/sign


# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())

    #set up variables
    n = len(w_coeffs)
    imaginary = []
    real = []
    abs = []
    rel = []

    #sort wilkinson roots
    w_roots = np.sort(w_roots)

    for _ in range(100):
        #randomly perturb the true coefficients of the wilinson polynomial
        r_i = np.random.normal(1, 1e-10, n)

        #calculate new roots and sort
        new_roots = np.roots(np.poly1d(w_coeffs*r_i))  
        new_roots = np.sort(new_roots)

        #get imaginary and real parts
        imaginary.append(new_roots.imag)
        real.append(new_roots.real)

        #calculate abs and rel conditioning number
        abs.append(la.norm(new_roots - w_roots, np.inf) / la.norm(r_i, np.inf))
        rel.append(abs[_]* la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf))

    #plot perturbed and original roots
    plt.scatter(real, imaginary, marker=",",s=1, label= "Perturbed")
    plt.scatter(w_roots.real, w_roots.imag, label= "Original")
    plt.xlabel("Real Parts of Roots")
    plt.ylabel("Imaginary Parts of Roots")
    plt.title("Roots")
    plt.legend()
    plt.show()


    #calculate averages of absolute and relative conditioning numbers and return
    return np.mean(abs), np.mean(rel)



# Helper function
def reorder_eigvals(orig_eigvals, pert_eigvals):
    """Reorder the perturbed eigenvalues to be as close to the original eigenvalues as possible.
    
    Parameters:
        orig_eigvals ((n,) ndarray) - The eigenvalues of the unperturbed matrix A
        pert_eigvals ((n,) ndarray) - The eigenvalues of the perturbed matrix A+H
        
    Returns:
        ((n,) ndarray) - the reordered eigenvalues of the perturbed matrix
    """
    n = len(pert_eigvals)
    sort_order = np.zeros(n).astype(int)
    dists = np.abs(orig_eigvals - pert_eigvals.reshape(-1,1))
    for _ in range(n):
        index = np.unravel_index(np.argmin(dists), dists.shape)
        sort_order[index[0]] = index[1]
        dists[index[0],:] = np.inf
        dists[:,index[1]] = np.inf
    return pert_eigvals[sort_order]

# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    #calculate real and imaginary parts of perturbation H
    reals = np.random.normal(0, 1e-10, np.shape(A))
    imags = np.random.normal(0, 1e-10, np.shape(A))
    H = reals + 1j*imags

    A_2 = la.norm(A, ord=2)
    H_2 = la.norm(H, ord=2)

    #compute eigenvalues of A and A+H
    A_eigs = la.eigvals(A)
    AH = A+H
    AH_eigs= la.eig(AH)[0]

    #reorder the perturbed evals
    AH_eigs = reorder_eigvals(A_eigs, AH_eigs)

    #compute relative and absolute condition number
    abs = la.norm((A_eigs-AH_eigs), ord=2)/H_2
    rel = abs*A_2/la.norm(A_eigs, ord=2)

    return abs, rel


# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    #isolate domains for linspaces to traverse over
    x_trav = np.linspace(domain[0], domain[1], res)
    y_trav = np.linspace(domain[2], domain[3], res)
    x_grid, y_grid = np.meshgrid(x_trav, y_trav)

    #create relative conditioning number array
    relative = np.zeros((res, res))

    #grab all relative conditioning numbers over grid
    for i, j in list(np.ndindex(res, res)):
        x, y = x_grid[i, j], y_grid[i, j]
        A = [[1, x], [y, 1]]
        abs, rel = eig_cond(A)
        relative[i, j] = rel

    #plot
    plt.pcolormesh(x_trav, y_trav, relative, cmap='gray_r', shading="auto")
    plt.colorbar()
    plt.show()



# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    #solve for the coefficients of polynomial of degree n that best fits the stability data
    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n+1)

    #use la.inv()
    c_inv = la.inv(A.T@A)@A.T@yk

    #use la.qr()
    Q, R = la.qr(A, mode="economic")
    c_qr = la.solve_triangular(R, Q.T@yk)

    #plot resulting polynomisal together with raw data points
    poly_inv = np.polyval(c_inv, xk)
    poly_qr = np.polyval(c_qr, xk)
    plt.plot(xk, poly_inv, label= "Inverse method")
    plt.plot(xk, poly_qr, label= "QR method")
    plt.scatter(xk, yk, marker=".", label="Raw Data")
    plt.ylim((0,25))

    plt.legend()
    plt.show()

    #return forward error of both approximations
    return la.norm((A@c_inv - yk), ord=2), la.norm((A@c_qr - yk), ord=2)


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    x = sy.symbols("x")
    I = lambda n: (x**int(n) * sy.exp(x-1))
    sy_I, ap_I = [], []
    domain = np.arange(5, 51, 5)

    for n in domain:
        sy_I.append(float(sy.integrate(I(n), (x, 0, 1))))
        ap_I.append((-1**n)*(sy.subfactorial(n) - (sy.factorial(n)/np.e)))
    
    sy_I, ap_I = np.array(sy_I), np.array(ap_I)
    rel_error = np.abs(sy_I-ap_I)/np.abs(sy_I)
    plt.semilogy(domain, rel_error, label= "relative errors")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print(prob2())