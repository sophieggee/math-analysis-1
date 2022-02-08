# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Sophie Gee>
<Section 2>
<1/7/22>
"""
import numpy as np
import numpy.linalg as la
from scipy import stats
from matplotlib import pyplot as plt

# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    #Get N random poiints in the n-d domain
    points = np.random.uniform(-1, 1, (N, n))

    #Determine how many points are within the circle
    lengths = la.norm(points, axis = 0)
    num_within = np.count_nonzero(lengths < 1)

    #Estimate volume
    return 2**n * (num_within / N)




# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    #Get N random poiints in the [a, b] domain
    points = np.random.uniform(a, b, N)

    #using 6.2, estimate integral
    est = ((b - a) / N) * np.sum([f(point) for point in points])

    return est


# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    n = len(mins)

    #Get N random poiints in the n-dimensional [0, 1] domain, then shift
    points = np.random.uniform(0, 1, (N, n))
    mins, maxs = np.array(mins), np.array(maxs)
    points = (maxs - mins) * points + mins

    #calulate average
    average = sum([f(points[i]) for i in range(N)]) / N

    V = np.prod(maxs - mins)

    return V * average


# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    #define f and Omega to call problem 3
    f = lambda x: (1/((2*np.pi)**(x.size/2)))*np.exp(-1 * (x @ x) / 2)
    Omega = [[-3/2, 0, 0, 0], [3/4, 1, 1/2, 1]]

    
    #the distribution has mean 0 and covariance I (the nxn identity)
    means, cov = np.zeros(4), np.eye(4)

    #compute integral
    F = stats.mvn.mvnun(Omega[0], Omega[1], means, cov)[0]

    #get 20 integer values of N roughly uniformly spaced 
    domain = np.logspace(1, 5, 20, dtype=int)

    #call problem 3
    rel_err = [(abs(F - mc_integrate(f, Omega[0], Omega[1], n)))/abs(F) for n in domain]

    #plot the relative error against the sample size N on log-log sclae
    plt.loglog(domain, rel_err, label= "Relative Error")
    plt.loglog(domain, 1/np.sqrt(domain), label= "1/sqrt(N)")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    prob4()
