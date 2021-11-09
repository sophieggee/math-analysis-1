# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""
from numpy.core.function_base import linspace
from numpy.core.numeric import full
from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt
from imageio import imread

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    #compute eigvals and vecs of AhA
    eigvals, eigvecs = la.eig(A.conj().T@A)

    #compute sing vals, sort from Greatest to Least
    
    sings = np.sqrt(eigvals)
    desc = sings.argsort()[::-1]
    sings = np.array([sings[i] for i in desc])
    eigvecs = np.array([eigvecs[i] for i in desc])

    #compute rank of A
    rank = 0
    for i in eigvals:
        if abs(i)>tol:
            rank+=1
    
    #keep only positive values
    sings=sings[:rank+1]
    eigvecs = eigvecs[:,:rank+1]
    
    #construct U
    U_1 = A@eigvecs/sings

    return U_1, sings, eigvecs.conj().T



# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    #define the circle matrix S
    domain = linspace(start=0,stop=2*np.pi,num=200)
    S = np.zeros((2,200))
    S[0,:] = np.cos(domain)
    S[1,:] = np.sin(domain)

    #define E
    E = np.array([[1,0,0],[0,0,1]])

    #compute svd of A
    U,s,V_h = la.svd(A)
    s = np.diag(s)

    #plot S against E
    g1 = plt.subplot(221)
    g1.plot(S[0,:],S[1,:])
    g1.plot(E[0,:], E[1,:])
    g1.axis("equal")

    #plot VhS and VhE
    VhS = V_h@S
    VhE = V_h@E
    g2 = plt.subplot(222)
    g2.plot(VhS[0,:],VhS[1,:])
    g2.plot(VhE[0,:], VhE[1,:])
    g2.axis("equal")

    #plot sigmaVhS and sigmaVhE
    sigVhS = s@V_h@S
    sigVhE = s@V_h@E
    g3 = plt.subplot(223)
    g3.plot(sigVhS[0,:],sigVhS[1,:])
    g3.plot(sigVhE[0,:], sigVhE[1,:])
    g3.axis("equal")

    #plot usigvhS and usigvhE
    UsigVhS = U@s@V_h@S
    UsigVhE = U@s@V_h@E
    g4 = plt.subplot(224)
    g4.plot(UsigVhS[0,:],UsigVhS[1,:])
    g4.plot(UsigVhE[0,:], UsigVhE[1,:])
    g4.axis("equal")

    plt.show()




# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    #collect the shape of A
    m,n = np.shape(A)

    #compute the compact SVD of A
    U,sig,V_h = la.svd(A, full_matrices=False)

    #raise appropriate value error
    if s > len([s for s in sig if abs(s)>0]):
        raise ValueError("Num of sing vals < s")

    #find truncated versions at s
    U_trunc = U[:,:s]
    V_h_trunc = V_h[:s,:]
    sig = np.diag(sig[:s])

    #collect lengths
    U_len = m*s
    V_h_len = s*n
    total_len = U_len+V_h_len+s

    return(U_trunc@sig@V_h_trunc, total_len)



# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    #compute the compact SVD of A
    U,sig,V_h = la.svd(A, full_matrices=False)

    #collect only the sigs less than error val
    SIG = [s for s in sig if s<err]
    if len(SIG) == 0:
        raise ValueError("A cannot be approximated within the tolerance by a matrix of lesser rank")
    
    #find sigmaS+1, and s as well
    sig_s_1 = SIG[np.argmax(SIG)]
    s = list(sig).index(sig_s_1)-1

    return(svd_approx(A, s))


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    image = imread(filename)/255

    #check if gray-scale, handle accordingly
    if len(np.shape(image)) == 2:
        num = np.shape(image)[0]*np.shape(image)[1]
        image_app, app_num = svd_approx(image,s)

        g1 = plt.subplot(121)
        g1.imshow(image, cmap="gray")
        plt.axis("off")

        g2 = plt.subplot(122)
        g2.imshow(image_app, cmap="gray")
        plt.axis("off")

    #check if color image, handle accordingly
    elif len(np.shape(image)) == 3:
        num = np.shape(image)[0]*np.shape(image)[1]*np.shape(image)[2]
        red_layer = image[:,:,0]
        green_layer = image[:,:,1]
        blue_layer = image[:,:,2]

        #calculate approx of each, and clip each
        approx_red, r = svd_approx(red_layer, s)
        approx_red = np.clip(approx_red,0,1)
        approx_green, g = svd_approx(green_layer, s)
        approx_green = np.clip(approx_green,0,1)
        approx_blue, b = svd_approx(blue_layer, s)
        approx_blue = np.clip(approx_blue,0,1)

        image_app = np.dstack((approx_red,approx_green,approx_blue))
        app_num = r+b+g

        g1 = plt.subplot(121)
        g1.imshow(image)
        plt.axis("off")

        g2 = plt.subplot(122)
        g2.imshow(image_app)
        plt.axis("off")

    plt.suptitle(f"Original image takes {num} entries, Compressed image takes {app_num} entries.")
    plt.show()

if __name__ == "__main__":
    compress_image("hubble.jpg", 20)