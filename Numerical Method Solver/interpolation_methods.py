import numpy as np

def newton_forward_interpolation(x_vals, y_vals, x):
    """
    Newton Forward Interpolation method.
    x_vals: list of x values
    y_vals: list of y values
    x: point where interpolation is needed
    """
    n = len(x_vals)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_vals

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i+1][j-1] - diff_table[i][j-1]

    h = x_vals[1] - x_vals[0]
    p = (x - x_vals[0]) / h
    result = y_vals[0]

    term = 1
    for j in range(1, n):
        term *= (p - j + 1) / j
        result += term * diff_table[0][j]

    return result



def newton_backward_interpolation(x_vals, y_vals, x):
    """
    Newton Backward Interpolation method.
    x_vals: list of x values
    y_vals: list of y values
    x: point where interpolation is needed
    """
    n = len(x_vals)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_vals

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            diff_table[i][j] = diff_table[i][j - 1] - diff_table[i - 1][j - 1]

    h = x_vals[1] - x_vals[0]
    p = (x - x_vals[-1]) / h
    result = y_vals[-1]

    term = 1
    for j in range(1, n):
        term *= (p + j - 1) / j
        result += term * diff_table[-1][j]

    return result
