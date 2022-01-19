# differentiation.py
"""Volume 1: Differentiation.
<Sophie Gee>
<volume 1>
<1/18/22>
"""

from turtle import color
import numpy as np
from matplotlib import pyplot as plt
from autograd import grad
from autograd import numpy as anp 
from autograd import elementwise_grad
import sympy as sy
import time
import random

# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""

    x = sy.symbols('x')
    df = sy.diff((sy.sin(x)+1)**sy.sin(sy.cos(x)), x)

    #lambdify
    lamb = sy.lambdify(x, df, "numpy")
    return lamb


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    #create x+h for input
    x1 = x+h
    return (f(x1) - f(x))/h

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    #create x+h, and x+2h for input
    x1 = x+h
    x2 = x+2*h
    return (-3*f(x) + 4*f(x1) - f(x2))/(2*h)

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    #create x-h for input
    x1 = x-h
    return (f(x) - f(x1))/h

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    #create x-h, and x-2h for input
    x1 = x-h
    x2 = x-2*h
    return (3*f(x) - 4*f(x1)  + f(x2))/(2*h)

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    #create x+h, x-h for input
    x1 = x+h
    x2 = x-h
    return (f(x1) - f(x2))/(2*h)

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    #create x+h, x-h, x-2h, x+2h for input
    x1 = x-2*h
    x2 = x-h
    x3 = x+h
    x4 = x+2*h
    return (f(x1) - 8*f(x2) + 8*f(x3) - f(x4))/(12*h)


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    #create the parameters necessary to plot graph
    h_1 = np.logspace(-8,0,9)
    f = lambda x: (np.sin(x) + 1)**np.sin(np.cos(x))
    df = prob1()
    err = lambda g,x,h: np.abs(df(x)-g(f,x,h))
    funct = [fdq1, fdq2, bdq1, bdq2, cdq2, cdq4]
    labels = ["Order 1 Forward", "Order 2 Forward", "Order 1 Backward",
    "Order 2 Backward", "Order 2 Centered", "Order 4 Centered"]
    y = []

    #two for loops, 1 for function, 1 for h values
    for j,f_1 in enumerate(funct):
        for i in h_1:
            y.append(err(f_1,x0,i))
        plt.loglog(h_1, y, marker=".", label = f"{labels[j]}")
        y = []

    #plot and show, use label
    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.show()


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    #load data and convert degrees to radians for alpha and beta
    data = np.load("plane.npy")
    data[:,1] = np.deg2rad(data[:,1])
    data[:,2] = np.deg2rad(data[:,2])

    #create array fo return values, sqrt(x'**2+y'**2), define x and y
    t_vals = []
    x = lambda a,b: (500*np.tan(b))/(np.tan(b)-np.tan(a))
    y = lambda a,b: (500*np.tan(b)*np.tan(a))/(np.tan(b)-np.tan(a))

    for i in range(8):
        #first order forward diff quotient
        if i == 0:
            x_p = x(data[i+1,1], data[i+1,2]) - x(data[i,1],data[i,2])
            y_p = y(data[i+1,1], data[i+1,2]) - y(data[i,1],data[i,2])
        #first order backward diff quotient
        elif i == 7:
            x_p = x(data[i,1], data[i,2]) - x(data[i-1,1],data[i-1,2])
            y_p = y(data[i,1], data[i,2]) - y(data[i-1,1],data[i-1,2])
        #second order centered diff quotient
        else:
            x_p = (x(data[i+1,1], data[i+1,2]) - x(data[i-1,1],data[i-1,2]))/2
            y_p = (y(data[i+1,1], data[i+1,2]) - y(data[i-1,1],data[i-1,2]))/2
        t_vals.append(np.sqrt(x_p**2+y_p**2))

    return t_vals


# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    n = len(x)
    #set up nxn identity matrix for ej
    E = np.eye(n)

    #create directional derivatives
    df_dj = lambda j: (f(x+h*E[j])-f(x-h*E[j]))/(2*h)

    return np.transpose([df_dj(j) for j in range(n)])


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """

    #check base cases
    if n == 0:
        return anp.ones_like(x)
    if n == 1:
        return x
    
    #return recursive relation
    else:
        return 2*x*cheb_poly(x,n-1) - cheb_poly(x,n-2)

def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    domain = anp.linspace(-1,1,1000)
    for n in range(5):
        gradient = elementwise_grad(cheb_poly)
        dg = gradient(domain,n)
        plt.plot(domain, dg, label=f"n = {n} derivative")
    plt.legend()
    plt.title("Gradients at Various N-Values")
    plt.show()


# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """
    f = lambda x: (anp.sin(x) + 1)**anp.sin(anp.cos(x))
    t_auto = []
    err_auto = []
    t_symp = []
    err_symp = []
    t_cdq4 = []
    err_cdq4 = []

    for i in range(N):
        #define random x0
        x0 = anp.random.random()

        #time sympy
        start = time.time()
        df = prob1()
        d = df(x0)
        end = time.time()
        t = end-start
        t_symp.append(t)
        err_symp.append(1e-18)

        #time diff quotients
        start = time.time()
        d1 = cdq4(f, x0, h=1e-5)
        end = time.time()
        t = end-start
        t_cdq4.append(t)
        error = np.abs(d-d1)
        err_cdq4.append(error)
        
        #time Autograd
        start = time.time()
        d_1 = grad(f)
        d2 = d_1(x0)
        end = time.time()
        t = end-start
        t_auto.append(t)
        error = np.abs(d-d2)
        err_auto.append(error)
    plt.loglog(t_symp, err_symp, ".",color="blue", label="SymPy", alpha=0.5, markeredgecolor='none')
    plt.loglog(t_cdq4, err_cdq4,".", color="orange", label="Difference Quotients", alpha=0.5, markeredgecolor='none')
    plt.loglog(t_auto, err_auto,".", color="green", label="Autograd", alpha=0.5, markeredgecolor='none')
    plt.xlabel("Computation Time (seconds)")
    plt.ylabel("Absolute Error")
    plt.title("Computation Time vs Error of Diff Methods")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    prob7()