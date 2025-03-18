import numpy as np

def gauss_eli(coeff, rhs):
    """Solves Ax = b using Gauss Elimination with error handling."""
    n = len(rhs)
    aug = np.column_stack((coeff, rhs))

    for r in range(n):
        if np.abs(aug[r][r]) < 1e-9:
            raise ValueError("Cannot proceed: Zero pivot or singular matrix.")

        for tr in range(r + 1, n):
            scale = aug[tr][r] / aug[r][r]
            aug[tr] -= scale * aug[r]

    sol = np.zeros(n)
    for r in range(n - 1, -1, -1):
        sol[r] = (aug[r][-1] - np.dot(aug[r][r+1:n], sol[r+1:n])) / aug[r][r]

    return sol

if __name__ == "__main__":
    try:
        var_count = int(input("Enter the number of variables: "))
        coeff = np.array([list(map(float, input(f"Equation {i+1} coefficients: ").split())) for i in range(var_count)])
        rhs = np.array([float(input(f"Constant for equation {i+1}: ")) for i in range(var_count)])

        solution = gauss_eli(coeff, rhs)
        print("\nSolution:", solution)
    except Exception as e:
        print(f"Error: {e}")
