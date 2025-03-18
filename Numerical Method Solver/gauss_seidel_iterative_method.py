import numpy as np

def is_diagonally_dominant(A):
    """Check if a matrix A is diagonally dominant."""
    n = len(A)
    for i in range(n):
        if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(n) if j != i):
            return False  # Not diagonally dominant
    return True



def gauss_seidel_iterative_method(A, b, x0=None, tol=1e-6, max_iterations=100):
    """Solves Ax = b using Gauss-Seidel with divergence handling."""
    n = len(A)
    if not is_diagonally_dominant(A):
        print("Warning: Matrix is not diagonally dominant. Gauss-Seidel may not converge.")

    if x0 is None:
        x0 = np.zeros(n)  # Default guess

    x = x0.copy()

    for it in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            sum_val = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_val) / A[i][i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new  # Converged

        x = x_new

    print("Warning: Gauss-Seidel did not converge within the given iterations.")
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

        solution = gauss_seidel_iterative_method(A, b, x0, tol, max_iter)
        print("\nSolution:", solution)
    except Exception as e:
        print(f"Error: {e}")
