import sympy as sp

def bisection_method(f_expr, a, b, tol=1e-6, max_iter=100):
    """Finds root using Bisection Method and tracks iteration history."""
    x = sp.symbols('x')
    f = sp.lambdify(x, f_expr)

    if f(a) * f(b) >= 0:
        raise ValueError("Invalid interval. f(a) and f(b) must have opposite signs.")

    iter_history = [(a + b) / 2]  # Store midpoints

    for i in range(max_iter):
        c = (a + b) / 2
        iter_history.append(c)

        if abs(f(c)) < tol:
            return c, iter_history  # Return root and iteration history

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return c, iter_history  # Return last computed root and iteration history

if __name__ == "__main__":
    try:
        f_expr = sp.sympify(input("Enter the function in terms of x: "))
        a = float(input("Enter the lower bound (a): "))
        b = float(input("Enter the upper bound (b): "))

        tol_input = input("Enter tolerance (optional, default=1e-6): ")
        tol = float(tol_input) if tol_input else 1e-6

        max_iter_input = input("Enter max iterations (optional, default=100): ")
        max_iter = int(max_iter_input) if max_iter_input else 100

        root, history = bisection_method(f_expr, a, b, tol, max_iter)
        print(f"Root found at x = {root}")

    except Exception as e:
        print(f"Error: {e}")
