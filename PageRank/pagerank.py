# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Sophie Gee>
<section 2>
<3/1/22>
"""
import numpy as np
import numpy.linalg as la
import networkx as nx
from itertools import combinations

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        n = len(A)

        #traverse through n columns to check if all zeros, then change to ones
        for col in range(n):
            if np.allclose(A[:, col], np.zeros((n,))):
                A[:, col] = np.ones((n,))
            #normalize
            A[:, col] = A[:, col] / np.sum(A[:, col])
        self.A = A
        self.n = len(A)

        #check labels and save as attribute
        if labels == None:
            self.labels = [i for i in range(n)]
        elif len(labels) != n:
            raise ValueError("Number of labels must be equal to the number of nodes in the graph.")
        else:
            self.labels = labels
        


    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #solve Ax = b
        b = ((1 - epsilon)/self.n)*np.ones((self.n,))
        A = np.eye(self.n) - (epsilon*self.A)
        A_inv = la.inv(A)

        #return A_inv@b
        p = A_inv@b

        dictionary = {self.labels[i]: p[i] for i in range(self.n)}

        return dictionary

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #calculate E and B
        E = np.ones((self.n, self.n))
        B = (epsilon*self.A) + ((1 - epsilon)/self.n)*E
        vals, vecs = la.eig(B)

        #get the eigenvalue index of eigval 1
        ind = np.argmax(vals)
        p = vecs[:, ind]

        #normalize
        p = p/np.sum(p)

        #calculate dictionary
        dictionary = {self.labels[i]: p[i] for i in range(self.n)}
        return dictionary

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #instantiate initial variables before iteration
        p_0 = np.array([1/self.n for i in range(self.n)])
        iters = 0
        eA = epsilon * self.A
        eps_n = ((1-epsilon)/self.n)*np.ones((self.n,))

        #start iteration
        while iters < maxiter:
            p_k = eA@p_0 + eps_n

            #check if tolerance is met
            if la.norm(p_k - p_0, ord=1) < tol:
                dictionary = {self.labels[i]: p_k[i] for i in range(self.n)}
                return dictionary
            else:
                p_0 = p_k

        #create a dictionary
        dictionary = {self.labels[i]: p_k[i] for i in range(self.n)}
        return dictionary
# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    vals, keys = zip(*sorted(zip(d.values(), d.keys())))
    return list(keys[::-1])
    


# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    with open(filename, 'r') as f:
        copy = f.read().strip()
    contents = copy.replace("\n", "/").split("/")
    contents = sorted(set(contents))

    #get the lengths of the dictionary
    n = len(contents)
    dictionary = {j:i for i,j in enumerate(contents)}

    A = np.zeros((n, n))
    for line in copy.split("\n"):
        sites = line.split("/")
        for site in sites[1:]:
            A[dictionary[site], dictionary[sites[0]]] += 1
    graph = DiGraph(A, contents)
    ranks = graph.itersolve(epsilon=epsilon)
    return get_ranks(ranks)



# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    with open(filename, 'r') as f:
        copy = f.read().strip()
    contents = copy.replace("\n", ",").split(",")[2:]
    contents = sorted(set(contents))

    #get the lengths of the dictionary
    n = len(contents)
    dictionary = {j:i for i,j in enumerate(contents)}
    
    copy = copy.split("\n")
    A = np.zeros((n, n))

    for line in copy[1:]:
        schools = line.split(",")
        for school in schools[1:]:
            A[dictionary[schools[0]], dictionary[school]] += 1
    graph = DiGraph(A, contents)
    ranks = graph.itersolve(epsilon=epsilon)
    return get_ranks(ranks)


# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    with open(filename, encoding="utf-8") as f:
        movies = [line.split("/") for line in f.read().splitlines()]
    graph = nx.DiGraph()

    for movie in movies:
        #get each combo of ators, one listed before the other if higher paid
        combo = combinations(movie[1:], 2)
        for actor1, actor2 in combo:
            # create the edge if it doesn't exist
            if not graph.has_edge(actor2, actor1):
                graph.add_edge(actor2, actor1, weight=0)
            #increase weight accordingly
            graph[actor2][actor1]["weight"] += 1
    return get_ranks(nx.pagerank(graph, alpha=epsilon))
    

if __name__ == "__main__":
    graph = DiGraph(np.array([[0, 0, 0, 0], [1, 0, 1, 0], [1 , 0, 0, 1], [1, 0, 1, 0]], dtype=float))
    print(rank_actors(epsilon = .7)[:3])
