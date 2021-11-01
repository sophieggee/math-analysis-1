# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Sophie Gee>
<Section 3>
<10/26/21>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import cmath


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q,R = la.qr(A, mode="economic")
    Q_T= np.transpose(Q)
    return la.solve_triangular(R, (np.matmul(Q_T,b)))


# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    #load in data, separate by columns
    data = np.load("housing.npy")
    year = data[:,0]
    index = data[:,1]
    A = np.column_stack((year,np.ones(len(year))))
    b = index
    #find slope using least squares equation computed
    slope_vector = least_squares(A,b)
    plt.plot(year, index, '.', label= "Data Points")
    domain = np.linspace(0,20)

    #plot
    plt.plot(domain,((slope_vector[0]*domain)+slope_vector[1]), label = "Least Squares Line")
    plt.legend()
    plt.title("Housing Prices and Years")
    plt.show()
    

# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    #load in the data, separate it by columms
    data = np.load("housing.npy")
    year = data[:,0]
    b = data[:,1]

    #create domain

    domain = np.linspace(0,16)

    #create polynomials of degree 3,6,9, and 12 to adjust slopes

    poly3 = np.vander(year, 4)
    A3 = poly3
    coef3= la.lstsq(A3, b)[0]
    f3 = np.poly1d(coef3)
    g1 = plt.subplot(221)
    g1.plot(domain,f3(domain), label = "Degree 3")
    g1.plot(year, b, '.', label= "Data Points")
    plt.legend()

    poly6 = np.vander(year, 7)
    A6 = poly6
    coef6= la.lstsq(A6, b)[0]
    f6 = np.poly1d(coef6)
    g2 = plt.subplot(222)
    g2.plot(domain,f6(domain), label = "Degree 6")
    g2.plot(year, b, '.', label= "Data Points")
    plt.legend()

    poly9 = np.vander(year, 10)
    A9 = poly9
    coef9= la.lstsq(A9, b)[0]
    f9 = np.poly1d(coef9)
    g3 = plt.subplot(223)
    g3.plot(domain,f9(domain), label = "Degree 9")
    g3.plot(year, b, '.', label= "Data Points")
    plt.legend()

    poly12 = np.vander(year, 13)
    A12 = poly12
    coef12= la.lstsq(A12, b)[0]
    f12 = np.poly1d(coef12)
    g4 = plt.subplot(224)
    g4.plot(domain,f12(domain), label = "Degree 12")
    g4.plot(year, b, '.', label= "Data Points")
    
    #plot slopes of best fit in subplots
    plt.suptitle("Least Square Polynomials")
    plt.legend()
    plt.show()
    


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

    

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    ellipse = np.load("ellipse.npy")
    x = ellipse[:,0]
    y = ellipse[:,1]

    #create an A based on ellipse equation
    A = np.column_stack((x**2,x, x*y,y,y**2))
    coef = la.lstsq(A,np.ones(len(A)))[0]
    plot_ellipse(*tuple(coef))
    plt.scatter(x,y)
    plt.title("Ellipse Line of Least Squares")
    plt.show()


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    #find n dimension
    m,n = np.shape(A)
    x = np.random.random(n)
    x = x/la.norm(x)
    #iterate through max numbers
    for k in range(N):
        y = np.matmul(A,x)
        y= y/la.norm(y)
        if la.norm(y-x) < tol:
            break
        x = y
    #return eigvals and vector
    return np.matmul(np.transpose(x),np.matmul(A,x)),x


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    #find n dimension
    m,n = np.shape(A)
    S = la.hessenberg(A)
    #iterate thrpigh number to run the algorithm
    for k in range(N):
        Q,R = la.qr(S)
        S = np.matmul(R,Q)
    eigs = []
    i = 0
    #while we have not gone through whole matrix
    while i < n:
        #as long as i is not the last value in matrix
        if i == n-1:
            eigs.append(S[i,i])
            i+=1 
        #if below value is around 0
        elif abs(S[i+1,i]) < tol:
            eigs.append(S[i,i])
            i+=1
        #if below value is actual value
        elif abs(S[i+1,i]) >= tol:
            a = S[i,i]
            b = S[i,i+1]
            c = S[i+1,i]
            d = S[i+1,i+1]
            B = -a-d
            A_1 = 1
            C = a*d-b*c
            #compute the quadratic formula to find different values
            eig_pos = (-B + np.sqrt(B**2-(4*C)))/2
            eig_neg = (-B - np.sqrt(B**2-(4*C)))/2
            eigs.append(eig_neg)
            eigs.append(eig_pos)
            i+=2
    return eigs
