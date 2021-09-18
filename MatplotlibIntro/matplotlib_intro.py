# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Sophie Gee>
<section 3>
<09/12/21>
"""
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    matrix_n = np.random.normal(size=(n,n)) #random matrix A with values drawn from the standard normal distribution
    mean = np.mean(matrix_n, axis=1) #mean value of each row
    variance = np.var(mean) #variance of these means
    return variance

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    array = [] #create array for results
    for n in range(100, 1001, 100):
        array += [var_of_means(n)] #find and insert results
    x = np.linspace(100, 1000, 10) #create x axis
    plt.plot(x,array) #plot
    plt.show() #show the plot


# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    neg_2_pi = -2*np.pi
    pos_2_pi = 2*np.pi

    x = np.linspace(neg_2_pi, pos_2_pi, 100) #creating a suitable domain
    y_sin = np.sin(x)
    y_cos = np.cos(x)
    y_arctan = np.arctan(x)
    plt.plot(x,y_sin) #plotting sin
    plt.show()
    plt.plot(x,y_cos) #plotting cos
    plt.show()
    plt.plot(x,y_arctan) #plotting arctan
    plt.show()


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    x1 = np.linspace(-2,1,100, endpoint=False) #creating a domain with an asymptot at 1
    y1 = 1/(x1-1) #plugging in x to the function
    plt.plot(x1,y1, 'm--', linewidth=4) #plotting line with visual specifications 
    x2 = np.linspace(1.00001,6,100) #creating a domain with an asymptot at 1
    y2 = 1/(x2-1) #plugging in x to the function
    plt.plot(x2,y2, 'm--', linewidth=4) #plotting line with visual specifications 
    plt.xlim(-2,6) #setting limits
    plt.ylim(-6,6)
    plt.show()

# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    ax1 = plt.subplot(221) #setting to top right corrner
    ax1.axis([0, 2*np.pi, -2, 2])
    x1 = np.linspace(0, 2*np.pi, 100) #creating a suitable domain
    y1_sin = np.sin(x1) 
    ax1.plot(x1, y1_sin, 'g')
    ax1.set_title("sin(x)", fontsize=10)
    
    ax2 = plt.subplot(222) #setting to top left corner
    ax2.axis([0, 2*np.pi, -2, 2])
    x2 = np.linspace(0, 2*np.pi, 100) #creating a suitable domain
    y2_sin = np.sin(2*x2)  #sin(2x)
    ax2.plot(x2, y2_sin, 'r--')
    ax2.set_title("sin(2x)", fontsize=10) #setting title
    
    ax3 = plt.subplot(223) #setting to bottom right corner
    ax3.axis([0, 2*np.pi, -2, 2])
    x3 = np.linspace(0, 2*np.pi, 100) #creating a suitable domain
    y3_sin = 2*np.sin(x3)  #2sin(x)
    ax3.plot(x3, y3_sin, 'b--')
    ax3.set_title("2sin(x)", fontsize=10) #setting title
    
    ax4 = plt.subplot(224) #setting to bottom left corner
    ax4.axis([0, 2*np.pi, -2, 2])
    x4 = np.linspace(0, 2*np.pi, 100) #creating a suitable domain
    y4_sin = 2*np.sin(2*x4)  #2sin(2x)
    ax4.plot(x4, y4_sin, 'm:')
    ax4.set_title("2sin(2x)", fontsize=10) #setting title
    plt.suptitle("Sin Graphs", fontsize = 15) #setting overall title
    plt.show()


# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    car_crash_data = np.load('FARS.npy') #loading in data
    hours = car_crash_data[:,0] 
    longitude = car_crash_data[:,1]
    latitude = car_crash_data[:,2]

    plt.plot(longitude,latitude, 'k,') #plotting the longitude against latitude for car crashes-
    plt.axis("equal") #setting the axes as scaled equal
    plt.xlabel('Longitude of Crash') #labeling the axes
    plt.ylabel('Latitude of Crash')
    plt.show()

    plt.hist(hours, bins=24, range=[0, 24]) #plotting a histogram with hours as bins
    plt.xlabel('Hour of Crash') #labeling the axes
    plt.ylabel('Number of Crashes Countrywide')
    plt.xlim(0,24)
    plt.show()

# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    neg_2_pi = -2*np.pi
    pos_2_pi = 2*np.pi
    x = np.linspace(neg_2_pi, pos_2_pi, 100) #setting x range
    y = x.copy() #setting y range
    X, Y = np.meshgrid(x, y) #create matrix
    G = np.sin(X)*np.sin(Y)/(X*Y) #operate on matrix

    plt.subplot(121) #create heat map
    plt.pcolormesh(X, Y, G, cmap="Pastel1")
    plt.colorbar() #include colorbar
    plt.xlim(neg_2_pi, pos_2_pi)
    plt.ylim(neg_2_pi, pos_2_pi)

    plt.subplot(122) #create contour map
    plt.contour(X, Y, G, 100, cmap="RdPu")
    plt.colorbar() #include colorbar
    plt.xlim(neg_2_pi, pos_2_pi)
    plt.ylim(neg_2_pi, pos_2_pi)

    plt.show()
