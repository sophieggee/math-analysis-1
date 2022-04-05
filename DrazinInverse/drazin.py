# drazin.py
"""Volume 1: The Drazin Inverse.
<Sophie Gee>
<Volume 1>
<April 5, 2022>
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse.csgraph import laplacian


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """

    #check to see if all three condiitons for a Drazin inverse hold

    if not np.allclose(A @ Ad, Ad @ A):
        return False

    if not np.allclose(np.linalg.matrix_power(A, k+1) @ Ad, 
                       np.linalg.matrix_power(A, k)):
        return False
    
    if not np.allclose(Ad @ A @ Ad, Ad):
        return False
    
    return True


# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """

    #get dimensions of A
    n = len(A)

    #use shur to get T1, Q1, k1
    T1, Q1, k1 = la.schur(A, sort=lambda x : abs(x) > tol)
    T2, Q2, k2 = la.schur(A, sort=lambda x: abs(x) <= tol)

    #build U and U inverse
    U = np.column_stack((Q1[:,:k1], Q2[:,:n-k1]))
    U_1 = la.inv(U)
    V = U_1 @ A @ U
    Z = np.zeros((n, n))

    #build block form
    if k1 != 0:
        M1 = la.inv(V[:k1,:k1])
        Z[:k1,:k1] = M1

    return U @ Z @ U_1



# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    n = len(A)
    I = np.eye(n)
    L = laplacian(A)
    R = np.zeros((n, n))

    #build effective resistence from node i to node j
    for j in range(n):
        Ljd = L.copy()
        Ljd[j] = I[j]
        Ljd = drazin_inverse(Ljd)
        R[:,j] = np.diag(Ljd)

    np.fill_diagonal(R, 0)
    return R

# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """

        #open the file and save them as persons
        with open(filename) as f:
            data = np.array([line.split(",") for 
                             line in f.read().splitlines()])
        persons = np.unique(np.ravel(data))

        #make dictionaries to go back and forth
        nodes_index = {team: i for i, team in enumerate(persons)}
        index_nodes = dict(enumerate(persons))


        n = len(persons)
        A = np.zeros((n, n))
        for p1, p2 in data:
            p1_index, p2_index = nodes_index[p1], nodes_index[p2]
            A[p1_index, p2_index] = 1
            A[p2_index, p1_index] = 1
        
        #set all the variables as state attributions
        self.persons = persons
        self.nodes_index = nodes_index
        self.index_nodes = index_nodes
        self.A = A
        self.R = effective_resistance(A)


    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        #assuming node is not none,
        if node is not None:
            if node not in self.nodes_index:
                raise ValueError(f"Node \"{node}\" is not in the network")
            i = self.nodes_index[node]
            Q_i = self.R[i][self.A[i] == 0]
            link, = np.ravel(np.where(self.R[i] == np.min(Q_i[Q_i != 0])))
            return self.index_nodes[link]
        
        #find links 1 and 2 for index of nodes
        Q = self.R[self.A == 0]
        link1, link2 = np.ravel(np.where(self.R == np.min(Q[Q != 0])))
        return self.index_nodes[link1], self.index_nodes[link2]


    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """

        #check to see if nodes are in the network
        if node1 not in self.nodes_index:
            raise ValueError(f"Node \"{node1}\" is not in the graph.")
        if node2 not in self.nodes_index:
            raise ValueError(f"Node \"{node2}\" is not in the graph.")
        
        i, j = self.nodes_index[node1], self.nodes_index[node2]
        self.A[i, j] = 1
        self.A[j, i] = 1
        self.R = effective_resistance(self.A)
