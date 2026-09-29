import numpy as np

def nest(d, c, x, b=None):
    """
    Evaluates a polynomial using Horneer's nested evaluation method.
    
    Parameters:
    d (int): Degree of the polynomial
    c (array-like): Coefficients list/array of length d+1
    x (float): The point at which to evaluate the polynomial
    b (array-like, optional): Base points list/array of length d. Defaults to zeros.
    """
    if b is None:
        b = np.zeros(d)

    #Start with the highest degree coefficient
    y=c[d]

    #Run the nesting algorithm backwards from d-1 down to 0
    for i in range (d - 1, -1, -1):
        y = y *  (x - b[i]) + c[i]

    return y