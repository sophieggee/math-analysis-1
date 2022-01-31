# newtons_method.py
"""Volume 1: Newton's Method.
<Sophie Gee>
<Volume 1>
<1/25/22>
"""

from autograd import grad
import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    converge = False

    if np.isscalar(x0):
    
        #iterate over maxiter
        for k in range(maxiter):
            #newton's equation
            x1 = x0 - alpha*(f(x0)/Df(x0))
            
            #check if the tolerance has been met
            if abs(x1-x0) < tol:
                converge = True
                break
            x0 = x1

        return x1, converge, k
    
    else:
        #iterate over maxiter
        for k in range(maxiter):
            #solve the linear system rather than computing Df at each step
            y0 = la.solve(Df(x0), f(x0))
            x1 = x0 - alpha*y0

            #check norm rather than absolute value
            if la.norm(x1-x0) < tol:
                converge = True
                break
            x0 = x1
        return x1, converge, k

# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    #create f in terms of r, set to 0
    f = lambda r: P1*((1+r)**N1-1)-P2*(1-(1+r)**-N2)
    df = grad(f)
    #call newton's method
    r, c, k = newton(f, .1, df)

    return(r)

# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    #create an alpha linspace for 100 values between 0 and 1
    alpha = np.linspace(0.001, 1, 100)
    best_alpha = []
    for a in alpha:
        #set the key of alpha equal to num of iters, append to best_alpha
        k = newton(f, x0, Df, tol, maxiter, a)[2]
        best_alpha.append(k)
   
    #plot the alphas against the number of iterations   
    plt.plot(alpha, best_alpha)
    plt.xlabel("Alpha Value")
    plt.ylabel("Number of Iterations")
    plt.title("Alphas and Iterations")
    plt.show()

    #return lowest, most optimal alpha term
    return alpha[np.argmin(best_alpha)]

# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    #define functions
    f_1 = lambda x: (5*x[0]*x[1])-(x[0]*(1+x[1]))
    f_2 = lambda x: (-1*x[0]*x[1])+(1-x[1])*(1+x[1])
    f = lambda x: np.array([f_1(x), f_2(x)])

    #define derivatives
    df1_dx = lambda v: (4 * v[1]) - 1
    df1_dy = lambda v: 4 * v[0]
    df2_dx = lambda v: -v[1]
    df2_dy = lambda v: -v[0] - 2 * v[1]
    df = lambda v: np.array([[df1_dx(v), df1_dy(v)], [df2_dx(v), df2_dy(v)]])


    #create the rectangle to search within
    x_domain = np.linspace(-.25, 0, 100)
    y_domain = np.linspace(0, .25, 100)
    rect = np.meshgrid(x_domain, y_domain)
    pos = np.column_stack([np.ravel(x) for x in rect])


    #search through and test values in grid rectangle
    #check if convergence is allclose and condition upon this
    for x in x_domain:
        for y in y_domain:
            x0 = np.array([x, y])
            try:
                a1 = newton(f, x0, df)
                if  (np.allclose(a1[0], [0., 1.]) or
                    np.allclose(a1[0], [0., -1.])):
                    a55 = newton(f, x0, df, alpha=0.55)
                    if  (np.allclose(a55[0], [3.75, 0.25])):
                        return x0
                else:
                    continue
            except:
                pass
    return "Not Found"

# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    #construct the resxres grid X0
    a_domain = np.linspace(domain[0], domain[1], res)
    b_domain = np.linspace(domain[2], domain[3], res)
    X_real, X_imag = np.meshgrid(a_domain, b_domain)
    X_0 = X_real + 1j*X_imag

    #run newtons method on X_0 iters times, obtain xk
    for k in range(iters):
        #solve the linear system rather than computing Df at each step
        X_0 = X_0 - f(X_0)/Df(X_0)

    #create another resxres array Y
    Y = np.zeros((res, res))

    #set Y values
    for i, j in list(np.ndindex(res, res)):
            Y[i,j] = np.argmin([la.norm(z-X_0[i,j]) for z in zeros])
    
    #plot out the colormesh
    plt.pcolormesh(X_real, X_imag, Y, cmap='brg', shading = 'auto')
    plt.xlabel("Real Parts")
    plt.ylabel("Imaginary Parts")
    plt.title("Basins of attraction")
    plt.show()

if __name__ == "__main__":
    print(prob6())