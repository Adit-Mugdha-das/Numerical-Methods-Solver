import numpy as np

def jacobi_iterative_method(A, b, x0, tol=1e-6, max_iter=100, callback=None):
    """Solves Ax = b using Jacobi Iterative Method with optional iteration tracking."""
    n = len(A)
    x = np.copy(x0)
    iter_history = []

    for it in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            sum_val = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_val) / A[i][i]
        
        if callback:
            callback(np.copy(x_new))  # Store iteration values for visualization

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new

        x = x_new

    return x

if __name__ == "__main__":
    try:
        n = int(input("Enter the number of equations: "))
        A = np.array([list(map(float, input(f"Row {i+1}: ").split())) for i in range(n)])
        b = np.array(list(map(float, input("Enter the constant vector: ").split())))

        # Ask for optional initial guess
        x0_input = input("Enter initial guess X0 (optional, space-separated values, default: zero vector): ")
        x0 = np.zeros(len(b)) if not x0_input else np.array(list(map(float, x0_input.split())))

        # Ask for optional tolerance and max iterations
        tol_input = input("Enter tolerance (optional, default=1e-6): ")
        tol = float(tol_input) if tol_input else 1e-6

        max_iter_input = input("Enter max iterations (optional, default=100): ")
        max_iter = int(max_iter_input) if max_iter_input else 100

        solution = jacobi_iterative_method(A, b, x0, tol, max_iter)
        print("\nSolution:", solution)
    except Exception as e:
        print(f"Error: {e}")
