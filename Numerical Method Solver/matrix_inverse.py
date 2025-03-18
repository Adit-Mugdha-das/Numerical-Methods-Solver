import numpy as np

def matrix_inverse(matrix):
    """Computes the inverse of a matrix using row operations (Gauss-Jordan Method)."""
    n = len(matrix)
    aug = np.hstack((matrix, np.eye(n)))  # Augment with identity matrix

    # Perform Gauss-Jordan elimination
    for i in range(n):
        # Partial pivoting to avoid division by zero
        if np.abs(aug[i][i]) < 1e-9:  # If pivot is near zero, swap rows
            for k in range(i+1, n):
                if np.abs(aug[k][i]) > 1e-9:
                    aug[[i, k]] = aug[[k, i]]  # Swap rows
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted.")

        # Normalize the pivot row
        aug[i] /= aug[i][i]

        # Make all other rows 0 in the current column
        for j in range(n):
            if i != j:
                aug[j] -= aug[i] * aug[j][i]

    return aug[:, n:]  # Extract the inverse matrix
