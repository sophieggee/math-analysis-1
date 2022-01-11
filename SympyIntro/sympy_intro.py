# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
<Sophie Gee>
<lab vol 1>
<1/11/22>
"""
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from sympy import Matrix

# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """

    #define variables x and y
    x = sy.symbols('x')
    y = sy.symbols('y')

    #establish expression and return
    expression = sy.Rational(2, 5)*sy.exp(x**2-y)*sy.cosh(x+y)+sy.Rational(3, 7)*sy.log((x*y)+1)
    return expression


# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    #establish initial expression
    x, i, j = sy.symbols('x i j')
    expr = sy.product(sy.summation(j*(sy.sin(x)+sy.cos(x)),(j, i, 5)), (i,1,5))

    #simplify
    return sy.simplify(expr)


# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    #define initial expression
    x, n, y = sy.symbols('x n y')
    expr = sy.summation((x**n)/sy.factorial(n),(n, 0, N))

    #substitu=te in -y**2
    y_expr = expr.subs(x, -y**2)

    #set domain, define functions, plot
    domain = np.linspace(-2,2,1000)
    f = sy.lambdify(y, y_expr, "numpy")
    h = lambda x: np.exp(-x**2)

    plt.plot(f(domain),domain, label= "lambdify")
    plt.plot(h(domain), domain, label= "lambda")
    plt.title("Lambdifying two different ways")
    plt.legend()
    plt.show()


# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    #construct equation 2.2
    x, y, r, t = sy.symbols('x y r t')
    expr = 1-((x**2+y**2)**sy.Rational(7,2) + 18*x**5*y - 60*x**3*y**3  +  18*x*y**5)/ (x**2+y**2)**3

    #convert it to polar coordinates
    pol_expr = expr.subs({x:r*sy.cos(t), y:r*sy.sin(t)})

    #simplify result
    simp = sy.simplify(pol_expr)

    #solve it for r
    solutions = sy.solve(simp, r)

    #lambdify solution
    f = sy.lambdify(t,solutions[1], "numpy")

    domain = np.linspace(0,np.pi*2, 1000)
    plt.plot(f(domain)*np.cos(domain), f(domain)*np.sin(domain))
    plt.title("rose curve")
    plt.show()

# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    x, y, l = sy.symbols('x y l')

    #define matrix A, L, and compute eigvals
    A = sy.Matrix([ [x-y, x, 0],
                    [x, x-y, x],
                    [0, x, x-y]])

    L = sy.Matrix([ [l, 0, 0],
                    [0, l, 0],
                    [0, 0, l]])

    I = sy.Matrix([ [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    
    eigvals = sy.solve(sy.det(A-L), l)


    dict_of_eigs = {}
    #compute eigvectors
    for i in eigvals:
        eigvect = (A-I*i).nullspace()[0]
        dict_of_eigs[i] = eigvect

    return dict_of_eigs




# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    #create f(x)
    x = sy.symbols('x')

    f = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100
    funct = sy.lambdify(x,f,"numpy")

    #find critical points
    df = sy.diff(f,x)
    cp = sy.solve(df, x)

    #calculate double derivative at these points
    ddf = sy.diff(df,x)
    local = sy.lambdify(x, ddf, "numpy")

    #plug in critical points to calculate maxima and minima
    maxima = []
    minima = []
    for i in cp:
        if local(i) > 0:
            minima.append(i)
        elif local(i) < 0:
            maxima.append(i)

    domain = np.linspace(-5,5,1000)

    #plot initial polynomial and min and max
    plt.plot(domain, funct(domain))
    plt.plot(np.array(minima),funct(np.array(minima)), '.r', label="min")
    plt.plot(np.array(maxima),funct(np.array(maxima)), '.b', label="max")
    plt.title("local min and max of polynomial")
    plt.legend()
    plt.show()

    return set(minima), set(maxima)




# Problem 7
def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    #create integrand
    x,y,z,r,o,t,rad = sy.symbols('x y z r o t rad')

    f = (x**2 + y**2 + z**2)**2
    h = f.subs({x: r*sy.sin(o)*sy.cos(t), y: r*sy.sin(o)*sy.sin(t), z:r*sy.cos(o)})

    m = sy.Matrix([r*sy.sin(o)*sy.cos(t), r*sy.sin(o)*sy.sin(t), r*sy.cos(o)])
    J = sy.simplify(m.jacobian([r,t,o]).det())
    integrand = sy.simplify(h*-J)

    #symbollic radius function
    rad_f = sy.integrate(integrand, (r,0,rad), (t,0,2*np.pi), (o, 0, np.pi))

    r_lamb = sy.lambdify(rad, rad_f, "numpy")
    domain = np.linspace(0,3,1000)

    #plot the integral and return when r=2
    plt.plot(domain, r_lamb(domain))
    plt.title("Integrand")
    plt.show()

    return r_lamb(2)

if __name__ == "__main__":
    print(prob5())