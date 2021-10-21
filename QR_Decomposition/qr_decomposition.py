# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Sophie Gee>
<section 3>
<10/19/21>
"""
import numpy as np
from scipy import linalg as la


# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    m,n = np.shape(A) #store dimensions of A
    Q = np.copy(A)     #make a copy of A
    R = np.zeros((n,n)) #make matrix of 0's

    for i in range(n):
        R[i,i] = la.norm(Q[:,i]) #Normalize the ith column of Q.
        Q[:,i] = Q[:,i]/R[i,i]
        for j in range(i+1,n):
            R[i,j] = np.dot(np.transpose(Q[:,j]),Q[:,i])
            Q[:,j] = Q[:,j]-R[i,j]*Q[:,i] # Orthogonalize the jth column of Q.
    return Q, R
   


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    Q,R = la.qr(A, mode="economic")
    return np.prod(np.diag(R))


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    Q,R = la.qr(A, mode="economic")
    y = np.matmul(np.transpose(Q), b) #set y to Qtb
    x = np.zeros((len(y)))
    for k in range(len(y)-1,-1,-1):
        x[k] = (1/R[k,k])*(y[k]-sum(R[k,k+1:]*x[k+1:])) #returns new y
    return x


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    sign = lambda x: 1 if x >= 0 else -1 #make sign function
    m,n = np.shape(A)   #store dimensions of A
    R = np.copy(A) #make a copy of A
    Q = np.identity(m) #make an identity matrix of size m 
    for k in (range(n)): #iterate through n-1
        u = np.copy(R[k:,k]) #make a copy called u
        u[0]= u[0]+ sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        R[k:,k:] = R[k:,k:] - np.outer(2*u, np.matmul(np.transpose(u), R[k:,k:]))
        Q[k:,:] = Q[k:,:] - np.outer(2*u, np.matmul(np.transpose(u), Q[k:,:]))
    return np.transpose(Q), R

# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    sign = lambda x: 1 if x >= 0 else -1 #make sign function
    m,n = np.shape(A) #store dimensions of A
    H = np.copy(A) #make a copy of A
    Q = np.identity(m) #make an identity matrix of size m
    for k in range(n-2): #iterate through n-3
        u = np.copy(H[k+1:,k]) #make a copy called u
        u[0]= u[0] + sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        H[k+1:,k:] = H[k+1:,k:] - np.outer(2*u, np.matmul(np.transpose(u), H[k+1:,k:]))
        H[:,k+1:] = H[:,k+1:] - 2*np.outer(np.matmul(H[:,k+1:],u), np.transpose(u))
        Q[k+1:,:] = Q[k+1:,:] - np.outer(2*u, np.matmul(np.transpose(u), Q[k+1:,:]))
    return H, np.transpose(Q)

if __name__ == "__main__":
    A = np.array([[1,2],[2,5]])
    Q,R = qr_gram_schmidt(A)
    print(qr_gram_schmidt(A))
    print(np.matmul(Q,R))

    

    
