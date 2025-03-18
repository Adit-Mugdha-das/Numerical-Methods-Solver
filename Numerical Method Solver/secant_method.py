import sympy as sp

def secant_method(f_expr, x0, x1, tol=1e-6, max_iter=100):
    """Finds root using Secant Method and tracks iteration history."""
    x = sp.symbols('x')
    f = sp.lambdify(x, f_expr)

    iter_history = [x0, x1]  # Store initial guesses

    for i in range(max_iter):
        if abs(f(x1)) < tol:
            return x1, iter_history  # Return root and iteration history

        if f(x1) == f(x0):
            raise ValueError("Zero denominator encountered, method fails.")

        x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        iter_history.append(x_new)

        x0, x1 = x1, x_new  # Update guesses

    return x1, iter_history  # Return last computed root and iteration history

if __name__ == "__main__":
    try:
        f_expr = sp.sympify(input("Enter the function in terms of x: "))
        x0 = float(input("Enter the first guess (x0): "))
        x1 = float(input("Enter the second guess (x1): "))

        tol_input = input("Enter tolerance (optional, default=1e-6): ")
        tol = float(tol_input) if tol_input else 1e-6

        max_iter_input = input("Enter max iterations (optional, default=100): ")
        max_iter = int(max_iter_input) if max_iter_input else 100

        root, history = secant_method(f_expr, x0, x1, tol, max_iter)
        print(f"Root found at x = {root}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
