def second_order_derivative(f, x, h=1e-5):
    """
    Computes second-order derivative using central difference.
    f: function
    x: point at which derivative is needed
    h: step size
    """
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
