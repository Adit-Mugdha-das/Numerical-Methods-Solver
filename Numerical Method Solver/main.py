import numpy as np
import sympy as sp
import runge_kutta
import gauss_eli
import gauss_jordan
import jacobi_iterative_method
import gauss_seidel_iterative_method
import lu_factorization
import matrix_inverse
import bisection_method
import false_position_method
import secant_method
import newton_raphson_method

def get_user_matrix():
    """Takes user input for a matrix and vector b."""
    n = int(input("Enter the number of equations (size of matrix): "))
    A = np.zeros((n, n))
    b = np.zeros(n)

    print("Enter the coefficient matrix (A) row by row:")
    for i in range(n):
        A[i] = list(map(float, input(f"Row {i+1}: ").split()))

    print("Enter the constant vector (b):")
    b = list(map(float, input().split()))

    return A, b

def get_user_function():
    """Takes user input for a function f(x)."""
    return sp.sympify(input("Enter the function in terms of x (e.g., x**3 - x - 2): "))

def main():
    while True:
        print("\n=== Numerical Methods Solver ===")
        print("1. Jacobi Iterative Method")
        print("2. Gauss-Seidel Iterative Method")
        print("3. LU Factorization")
        print("4. Gauss Elimination")
        print("5. Gauss-Jordan Elimination")
        print("6. Matrix Inversion")
        print("7. Runge-Kutta Method")
        print("8. Bisection Method")
        print("9. False Position Method")
        print("10. Secant Method")
        print("11. Newton-Raphson Method")
        print("12. Exit")

        choice = input("Choose a method (1-12): ")

        if choice == "1":
            A, b = get_user_matrix()
            x0 = np.zeros(len(b))
            solution = jacobi_iterative_method.jacobi_method(A, b, x0)
            print("Solution:", solution)

        elif choice == "2":
            A, b = get_user_matrix()
            x0 = np.zeros(len(b))
            solution = gauss_seidel_iterative_method.gauss_seidel(A, b, x0)
            print("Solution:", solution)

        elif choice == "3":
            A, _ = get_user_matrix()
            L, U = lu_factorization.lu_decomposition(A)
            print("Lower Triangular Matrix L:\n", L)
            print("Upper Triangular Matrix U:\n", U)

        elif choice == "4":
            A, b = get_user_matrix()
            solution = gauss_eli.gauss_solver(A, b)
            print("\nGauss Elimination Solution:")
            for i, res in enumerate(solution):
                print(f"x{i + 1} = {res}")

        elif choice == "5":
            A, b = get_user_matrix()
            solution = gauss_jordan.gauss_jordan_solver(A, b)
            print("\nGauss-Jordan Solution:")
            for i, res in enumerate(solution):
                print(f"x{i + 1} = {res}")

        elif choice == "6":
            matrix_inverse.get_user_input()

        elif choice == "7":
            runge_kutta.get_user_input()

        elif choice == "8":
            f_expr = get_user_function()
            a = float(input("Enter the lower bound (a): "))
            b = float(input("Enter the upper bound (b): "))
            bisection_method.bisection_method(f_expr, a, b)

        elif choice == "9":
            f_expr = get_user_function()
            a = float(input("Enter the lower bound (a): "))
            b = float(input("Enter the upper bound (b): "))
            false_position_method.false_position_method(f_expr, a, b)

        elif choice == "10":
            f_expr = get_user_function()
            x0 = float(input("Enter the first guess (x0): "))
            x1 = float(input("Enter the second guess (x1): "))
            secant_method.secant_method(f_expr, x0, x1)

        elif choice == "11":
            f_expr = get_user_function()
            x0 = float(input("Enter the initial guess (x0): "))
            newton_raphson_method.newton_raphson(f_expr, x0)

        elif choice == "12":
            print("Exiting the program. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number from 1 to 12.")

if __name__ == "__main__":
    main()
