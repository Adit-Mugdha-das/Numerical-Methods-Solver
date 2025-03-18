def simpsons_one_third_rule(f, a, b, n=100):
    """
    Simpson's 1/3 Rule for numerical integration.
    f: function to integrate
    a, b: integration limits
    n: number of subintervals (should be even)
    """
    if n % 2 == 1:
        raise ValueError("n must be even for Simpson's 1/3 rule.")

    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    y = [f(val) for val in x]

    result = y[0] + y[-1]
    for i in range(1, n, 2):
        result += 4 * y[i]
    for i in range(2, n-1, 2):
        result += 2 * y[i]

    return (h / 3) * result


def simpsons_three_eighth_rule(f, a, b, n=3):
    """
    Simpson's 3/8 Rule for numerical integration.
    f: function to integrate
    a, b: integration limits
    n: number of subintervals (should be a multiple of 3)
    """
    if n % 3 != 0:
        raise ValueError("n must be a multiple of 3 for Simpson's 3/8 rule.")

    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    y = [f(val) for val in x]

    result = y[0] + y[-1]
    for i in range(1, n, 3):
        result += 3 * y[i] + 3 * y[i+1]
    for i in range(3, n-1, 3):
        result += 2 * y[i]

    return (3 * h / 8) * result
