# linear_transformations.py
"""Volume 1: Linear Transformations.
<Sophie Gee>
<Vol 1 Lab section 3>
<09/21/21>
"""
import numpy as np
from random import random
from matplotlib import pyplot as plt
import time 


# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    streth_matrix=np.array([[a,0],[0,b]])
    return (streth_matrix@A)

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    shear_matrix = np.array([[1,a],[b,1]])
    return np.matmul(shear_matrix, A)

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    outside_constant = (1/(a**2+b**2))
    base_matrix = np.array([[a**2-b**2, 2*a*b],[2*a*b, b**2-a**2]])
    reflect_matrix = outside_constant*base_matrix
    return np.matmul(reflect_matrix, A)

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return np.matmul(rotate_matrix,A)

# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    initial_pe = [x_e, 0]
    t=np.linspace(0,T,200)
    P_E = np.vstack([rotate(initial_pe, omega_e*i) for i in t])
    print(P_E)
    vector_of_moon = [x_e-x_m, 0]
    P_M = np.vstack([rotate(vector_of_moon, omega_m*i) for i in t])
    P_of_M = P_M+P_E
    P_of_M = P_of_M.T
    plt.plot(P_E[:,0], P_E[:,1], label="Earth")
    plt.plot(P_of_M[0], P_of_M[1], label="Moon")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.show()



def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """

    domain = 2**np.arange(1,9)
    times = []
    for n in domain:
        A = random_matrix(n)
        x = random_vector(n)
        start = time.time()
        matrix_vector_product(A,x)
        times.append(time.time() - start)
    ax1 = plt.subplot(121)
    ax1.plot(domain, times, '.-', linewidth=2, markersize=15)
    ax1.set_title("Matrix-Vector Multiplication")
    ax1.set_xlabel("n", fontsize=14)
    ax1.set_ylabel("Seconds", fontsize=14)
    times2=[]
    for n in domain:
        A = random_matrix(n)
        B = random_matrix(n)
        start = time.time()
        matrix_matrix_product(A,B)
        times2.append(time.time() - start)
    ax2 = plt.subplot(122)
    ax2.plot(domain, times2, 'g.-', linewidth=2, markersize=15)
    ax2.set_title("Matrix-Matrix Multiplication")
    ax2.set_xlabel("n", fontsize=14)
    ax2.set_ylabel("Seconds", fontsize=14)
    plt.show()



# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    domain = 2**np.arange(1,9)
    times = []
    for n in domain:
        A = random_matrix(n)
        x = random_vector(n)
        start = time.time()
        matrix_vector_product(A,x)
        times.append(time.time() - start)
    times_1 = []
    for n in domain:
        A = random_matrix(n)
        B = random_matrix(n)
        start = time.time()
        matrix_matrix_product(A,B)
        times_1.append(time.time() - start)
    times_2=[]
    for n in domain:
        A = random_matrix(n)
        x = random_vector(n)
        start = time.time()
        np.dot(A,x)
        times_2.append(time.time() - start)
    times2=[]
    for n in domain:
        A = random_matrix(n)
        B = random_matrix(n)
        start = time.time()
        np.dot(A,B)
        times2.append(time.time() - start)
    ax1 = plt.subplot(121)
    ax1.plot(domain, times, '.-', linewidth=2, markersize=15, label="Matrix-Vector by Hand")
    ax1.plot(domain, times_1, '.-', linewidth=2, markersize=15, label="Matrix-Matrix by Hand")
    ax1.plot(domain, times_2, '.-', linewidth=2, markersize=15, label="Matrix-Vector Numpy")
    ax1.plot(domain, times2, '.-', linewidth=2, markersize=15, label="Matrix-Matrix Numpy")
    ax1.set_xlabel("n", fontsize=14)
    ax1.set_ylabel("Seconds", fontsize=14)
    ax1.legend()
    ax2 = plt.subplot(122)
    ax2.loglog(domain, times, '.-', basex=2, basey=2, lw=2)
    ax2.loglog(domain, times_1, '.-', basex=2, basey=2, lw=2)
    ax2.loglog(domain, times_2, '.-', basex=2, basey=2, lw=2)
    ax2.loglog(domain, times2, '.-', basex=2, basey=2, lw=2)
    ax2.set_xlabel("n", fontsize=14)
    ax2.set_ylabel("Seconds", fontsize=14)
    plt.show()

if __name__ == "__main__":
    prob4()
