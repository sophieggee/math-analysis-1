# standard_library.py
"""Python Essentials: The Standard Library.
<Sophie Gee>
<MATH 345>
<09/07/21>
"""
import calculator
from itertools import combinations

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    average = sum(
        L)/len(L)  # compute the average by dividing sum by number of entries
    return min(L), max(L), average

# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    #mutability experiment for int:
    num = 17
    new_num = num
    new_num += 1
    int_bool = (num == new_num)

    #mutability experiment for string:
    string = "hello world"
    new_string = string
    new_string += " hi girl"
    str_bool = (string == new_string)

    #mutability experiment for list:
    list = ["hello", "world", "hi", "girl"]
    new_list = list
    new_list += ["sophie"]
    list_bool = (list == new_list)

    #mutability experiment for tuple:
    tuple = ("hello", "world", "hi", "girl")
    new_tuple = tuple
    new_tuple += ("sophie",)
    tuple_bool = (tuple == new_tuple)

    #mutability experiment for set:
    set = {1, 2, 3}
    new_set = set
    new_set.add("sophie")
    set_bool = (set == new_set)

    return (f"int is mutable: {int_bool}", f"str is mutable: {str_bool}",
           f"list is mutable: {list_bool}", f"tuple is mutable: {tuple_bool}",
           f"set is mutable: {set_bool}")

print(prob2())


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt that are 
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    a_squared = calculator.product(a,a) #squaring side a
    b_squared = calculator.product(b,b) #squaring side b
    sum_of_squares = calculator.sum(a_squared,b_squared) #a squared plus b squared
    hypotenuse = calculator.sqrt(sum_of_squares) #calculating square root of above sum (hypotenuse equation)
    return hypotenuse

# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    list_of_sets = [] # creates empty list of sets
    for i in range(0, len(A)+1): # iterates through index of passed in iterable
        list_of_sets.append(list(combinations(A,i))) #appends multiple combinations that will make up powerset
    return list_of_sets

# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
