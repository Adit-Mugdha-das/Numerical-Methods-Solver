import sympy as sp

def newton_raphson_method(f_expr, x0, tol=1e-6, max_iter=100):
    """Finds root using Newton-Raphson Method and tracks iteration history."""
    x = sp.symbols('x')
    f = sp.lambdify(x, f_expr)
    df_expr = sp.diff(f_expr, x)
    df = sp.lambdify(x, df_expr)

    x_curr = x0
    iter_history = [x_curr]  # Store iterations

    for i in range(max_iter):
        if abs(f(x_curr)) < tol:
            return x_curr, iter_history  # Return root and iteration history

        if df(x_curr) == 0:
            raise ValueError("Derivative is zero, method fails.")

        x_curr -= f(x_curr) / df(x_curr)
        iter_history.append(x_curr)

    return x_curr, iter_history  # Return last computed root and iteration history

if __name__ == "__main__":
    try:
        f_expr = sp.sympify(input("Enter the function in terms of x: "))
        x0 = float(input("Enter the initial guess (x0): "))

        tol_input = input("Enter tolerance (optional, default=1e-6): ")
        tol = float(tol_input) if tol_input else 1e-6

        max_iter_input = input("Enter max iterations (optional, default=100): ")
        max_iter = int(max_iter_input) if max_iter_input else 100

        root, history = newton_raphson_method(f_expr, x0, tol, max_iter)
        print(f"Root found at x = {root}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
