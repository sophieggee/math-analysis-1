# profiling.py
"""Python Essentials: Profiling.
<Sophie Gee>
<Class>
<1/4/22>
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
from numba import jit, int64, double
from numpy import linalg as la
from time import time
from matplotlib import pyplot as plt

# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    
    #get all of the numbers in the triangle alone
    with open(filename, 'r') as myfile:
        data = [[int(n) for n in line.split()]
                        for line in myfile.readlines()]

    # go through triangle from bottom up, replacing each entry with largest sum
    i = len(data)-2
    while i >= 0:
        #make sure going through each row as well
        for j in range(len(data[i])):
            data[i][j] = data[i][j] + max(data[i+1][j], data[i+1][j+1])
        i -= 1
        
    return data[0][0] #first entry will have largest sum

# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = [2]
    current = 3
    count = 1 #create an iterable
    while count < N:
        isprime = True
        for i in primes_list: #check for divisors in primes_list
            if i*i > current:
                break    
            if current % i == 0:
                isprime = False
                break #break if not prime
        if isprime: 
            primes_list.append(current)
            count +=1 #iterate count iterable
        current += 2 #go through only odd numbers
    return primes_list


# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    return np.argmin(np.linalg.norm(A.T-x, axis=1)) #compute the matrix norm, row-wise of transpose of A-x and find the min


# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter_values = {letter: i+1 for i,letter in enumerate(alphabet)}
    name_val = lambda name: sum([letter_values[letter] for letter in name])
    total = sum([(i+1)*name_val(name) for i,name in enumerate(names)])
    return total


# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    fn_1=1
    fn_2=1
    #step through each fibonacci sequence using two yields
    yield fn_1
    yield fn_2
    while True:
        fn_1 += fn_2
        yield fn_1
        fn_2 += fn_1
        yield fn_2

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    for i, seq in enumerate(fibonacci()):
        if len(str(seq)) == N:
            return i+1
    
# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    ints = np.arange(2,N) #create a list of all of the integers from 2 to N
    while len(ints) > 0: #while ints is not empty
        num = ints[0]
        yield num
        ints = np.delete(ints, 0) #remove first entry
        ints = ints[np.where(ints % num != 0)] #remove other divisible entries



# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

#jit decorator important!
@jit(nopython=True, locals=dict(A=double[:,:], n=int64, product=double[:,:],
                                temporary_array=double[:], m=int64, total=double))
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    # Run matrix_power_numba first for it to compile, run it for 2**2
    matrix_power_numba(np.zeros((2, 2)), 2)
    
    # Time each function
    a1, a2, a3 = [], [], []
    for i in range(2, 8):
        m = 2 ** i
        A = np.random.random((m, m))
        
        #time takes to run matrix_power
        t1 = time()
        matrix_power(A, n)
        a1.append((m, time() - t1))
        
        #time takes to run matrix_power_numba
        t2 = time()
        matrix_power_numba(A, n)
        a2.append((m, time() - t2))
    
        #time takes to run np.la
        t3 = time()
        la.matrix_power(A, n)
        a3.append((m, time() - t3))
        
    plt.loglog(*zip(*a1), label="matrix_power function")
    plt.loglog(*zip(*a2), label="matrix_power_numba function")
    plt.loglog(*zip(*a3), label="np.la.matrix_power funciton")
    plt.xlabel("Time")
    plt.ylabel("Size of matrix")
    plt.title("Time Taken to run Functions vs Size")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print(name_scores_fast())
