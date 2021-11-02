# image_segmentation.py
"""Volume 1: Image Segmentation.
<Sophie Gee>
<section 3>
<11/2/21>
"""

import numpy as np
from numpy.core.fromnumeric import partition
from scipy import linalg as la
from imageio import imread
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    n,m = np.shape(A)
    D = np.zeros((n,n))
    for i in range(n):
        D[i,i] = np.sum(A, axis=0)[i]
    L = D-A
    return L

# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    L = laplacian(A)
    connect = 0
    eigs = np.real(la.eigvals(L))
    for eig in eigs:
        if abs(eig) < tol:
            connect += 1
            eig = 0
    return (connect, partition(eigs, 1)[1])


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        image = imread(filename)
        scaled = image/255
        self.im=scaled

        #if image is in color
        if len(np.shape(image)) == 3:
            brightness = np.mean(scaled,axis=2)

        #else, image is grayscale
        else:
            brightness = scaled

        #set brightness, then flatten it
        self.bright = brightness
        self.flat_bright = np.ravel(brightness)


    # Problem 3
    def show_original(self):
        """Display the original image."""
        if np.shape(self.im) == 3:
            plt.imshow(self.im)
        else:
            plt.imshow(self.im, cmap="gray")
        plt.axis("off")

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""

        #intitialize variables and set to empty matrices/arrays
        mn = len(self.flat_bright)
        A = sparse.lil_matrix((mn,mn))
        D = np.zeros(mn)
        B = self.flat_bright
        if len(np.shape(self.im)) == 3:
            m,n, _ = np.shape(self.im)
        else:
            m,n =np.shape(self.im)

        #go through each index of each pixel in the image
        for i in range(mn):
            #get array of Ji and distances between each i and j
            arr, dist = (get_neighbors(i, r, m, n))
            b = -abs((B[i]-B[arr]))/sigma_B2
            x = -(dist/sigma_X2)
            #set each weight array using the equation given
            weight = np.exp(b+x)
            #set A and D
            A[i, arr]= weight
            D[i] = np.sum(weight)
        
        #convert to diagonal matrix and sparse optimized matrix
        A = sparse.csc_matrix(A)

        return A,D

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        L = sparse.csgraph.laplacian(A)
        D1 = sparse.diags([1/np.sqrt(d) for d in D])
        DLD = D1@L@D1
        eval, evec = eigsh(DLD,which="SM", k=2)
        evec = evec[:,1]
        if len(np.shape(self.im)) == 3:
            m,n, _ = np.shape(self.im)
        else:
            m,n =  np.shape(self.im)
        reshaped = evec.reshape(m,n)
        mask = reshaped > 0
        return mask

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A, D = self.adjacency(r, sigma_B, sigma_X)
        mask = self.cut(A,D)
        if len(np.shape(self.im)) == 3:
            if np.shape(self.im)[2] > 3:
                mask = np.dstack((mask,mask,mask,mask))
            else:
                mask = np.dstack((mask,mask,mask))
    

        plt.subplot(131)
        plt.imshow(self.im)
        plt.axis("off")

        pos = self.im*mask
        plt.subplot(132)
        plt.imshow(pos)
        plt.axis("off")

        neg = self.im*~mask
        plt.subplot(133)
        plt.imshow(neg)

        plt.axis("off")

        plt.show()


if __name__ == '__main__':
    ImageSegmenter("dream_gray.png").segment()
    ImageSegmenter("dream.png").segment()
    ImageSegmenter("blue_heart.png").segment()
