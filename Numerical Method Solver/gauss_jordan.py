import numpy as np

def gauss_jordan(coeff, rhs):
    """Solves Ax = b using Gauss-Jordan Method with error handling."""
    n = len(rhs)
    aug = np.column_stack((coeff, rhs))

    for r in range(n):
        if np.abs(aug[r][r]) < 1e-9:
            raise ValueError("Cannot proceed: Zero pivot or singular matrix.")

        aug[r] /= aug[r][r]
        for tr in range(n):
            if tr != r:
                aug[tr] -= aug[tr][r] * aug[r]

    return aug[:, -1]

if __name__ == "__main__":
    try:
        var_count = int(input("Enter the number of variables: "))
        coeff = np.array([list(map(float, input(f"Equation {i+1} coefficients: ").split())) for i in range(var_count)])
        rhs = np.array([float(input(f"Constant for equation {i+1}: ")) for i in range(var_count)])

        solution = gauss_jordan(coeff, rhs)
        print("\nSolution:", solution)
    except Exception as e:
        print(f"Error: {e}")
