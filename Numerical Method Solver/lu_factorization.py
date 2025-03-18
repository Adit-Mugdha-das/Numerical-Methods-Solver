import numpy as np

def lu_factorization(A, b):
    """Performs LU Decomposition using Doolittle's method with error handling."""
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            sum_val = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_val

        if U[i][i] == 0:
            raise ValueError("Matrix is singular, LU decomposition cannot proceed.")

        L[i][i] = 1
        for k in range(i+1, n):
            sum_val = sum(L[k][j] * U[j][i] for j in range(i))
            L[k][i] = (A[k][i] - sum_val) / U[i][i]

    # Solve Ly = b using forward substitution
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]

    # Solve Ux = y using back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]

    return L, U, y, x  # Returning all computed values

if __name__ == "__main__":
    try:
        n = int(input("Enter the size of the square matrix (n x n): "))
        if n <= 0:
            raise ValueError("Matrix size must be a positive integer.")

        A = np.array([list(map(float, input(f"Row {i+1}: ").split())) for i in range(n)])
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")

        b = np.array(list(map(float, input("Enter the constant vector b: ").split())))

        if len(b) != n:
            raise ValueError("Size of b must match number of rows in A.")

        # Perform LU decomposition and solve the system
        L, U, y, x = lu_factorization(A, b)

        print("\nLower Triangular Matrix L:\n", L)
        print("\nUpper Triangular Matrix U:\n", U)
        print("\nIntermediate solution (y):", y)
        print("\nFinal solution (x):", x)

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
