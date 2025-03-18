from tkinter import *
from tkinter import messagebox
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import bisection_method
import false_position_method
import newton_raphson_method
import secant_method
import runge_kutta
import gauss_eli
import gauss_jordan
import gauss_seidel_iterative_method
import jacobi_iterative_method
import tkinter as tk
from tkinter import Toplevel, Text, Scrollbar
import lu_factorization
import matrix_inverse
import os
import pdfkit
import webbrowser
from interpolation_methods import newton_forward_interpolation
from integration_methods import simpsons_one_third_rule
from differentiation_methods import second_order_derivative
from integration_methods import simpsons_three_eighth_rule
from interpolation_methods import newton_backward_interpolation


root = Tk()
root.title("Numerical Methods Solver")
root.geometry("1200x700")
root.configure(bg="#1E1E1E")  # Dark mode

Label(root, text="Numerical Methods Solver", font=("Arial", 16, "bold"), bg="#1E1E1E", fg="white").pack(pady=10)

# Define button style
button_style = {
    "width": 30,
    "font": ("Arial", 10),
    "bg": "#333333",
    "fg": "white",
    "relief": "raised",
    "activebackground": "#444444",
    "activeforeground": "white"
}

# Create a Frame to hold all categories
main_frame = Frame(root, bg="#1E1E1E")
main_frame.pack(pady=10)

def open_help_window():
    url = "https://github.com/Adit-Mugdha-das/Numerical-Methods-Solver/edit/main/README.md"  # Replace with your actual URL
    webbrowser.open(url)  # Opens in the default web browser


# Function to create a popup window for user input dynamically
def get_input_and_run(method_name, method_func, method_type):
    popup = Toplevel(root)
    popup.title(method_name)
    popup.geometry("400x500")

    Label(popup, text=f"Enter inputs for {method_name}", font=("Arial", 12)).pack(pady=10)

    entries = {}

    if method_type in ["root_finding", "newton_raphson", "secant", "runge_kutta"]:
        Label(popup, text="Function (e.g., x**3 - x - 2 or dy/dx in terms of x, y):").pack()
        func_entry = Entry(popup, width=30)
        func_entry.pack()
        entries["func"] = func_entry

    if method_type == "root_finding":
        Label(popup, text="Lower Bound (a):").pack()
        a_entry = Entry(popup, width=15)
        a_entry.pack()
        entries["a"] = a_entry

        Label(popup, text="Upper Bound (b):").pack()
        b_entry = Entry(popup, width=15)
        b_entry.pack()
        entries["b"] = b_entry

        # Add optional tolerance input
        Label(popup, text="Tolerance (optional, default=1e-6):").pack()
        tol_entry = Entry(popup, width=15)
        tol_entry.pack()
        entries["tol"] = tol_entry

        # Add optional max iterations input
        Label(popup, text="Max Iterations (optional, default=100):").pack()
        max_iter_entry = Entry(popup, width=15)
        max_iter_entry.pack()
        entries["max_iter"] = max_iter_entry

        Button(popup, text="Graph Visualization", command=lambda: visualize_root_finding(method_name, entries)).pack(pady=5)
    
    elif method_type == "newton_raphson":
        Label(popup, text="Initial Guess (x0):").pack()
        x0_entry = Entry(popup, width=15)
        x0_entry.pack()
        entries["x0"] = x0_entry

        # Add optional tolerance input
        Label(popup, text="Tolerance (optional, default=1e-6):").pack()
        tol_entry = Entry(popup, width=15)
        tol_entry.pack()
        entries["tol"] = tol_entry

        # Add optional max iterations input
        Label(popup, text="Max Iterations (optional, default=100):").pack()
        max_iter_entry = Entry(popup, width=15)
        max_iter_entry.pack()
        entries["max_iter"] = max_iter_entry
        Button(popup, text="Graph Visualization", command=lambda: visualize_root_finding(method_name, entries)).pack(pady=5)

    elif method_type == "secant":
        Label(popup, text="First Guess (x0):").pack()
        x0_entry = Entry(popup, width=15)
        x0_entry.pack()
        entries["x0"] = x0_entry

        Label(popup, text="Second Guess (x1):").pack()
        x1_entry = Entry(popup, width=15)
        x1_entry.pack()
        entries["x1"] = x1_entry

        # Add optional tolerance input
        Label(popup, text="Tolerance (optional, default=1e-6):").pack()
        tol_entry = Entry(popup, width=15)
        tol_entry.pack()
        entries["tol"] = tol_entry

        # Add optional max iterations input
        Label(popup, text="Max Iterations (optional, default=100):").pack()
        max_iter_entry = Entry(popup, width=15)
        max_iter_entry.pack()
        entries["max_iter"] = max_iter_entry
        Button(popup, text="Graph Visualization", command=lambda: visualize_root_finding(method_name, entries)).pack(pady=5)
        


    elif method_type == "runge_kutta":
        Label(popup, text="Initial x (x0):").pack()
        x0_entry = Entry(popup, width=15)
        x0_entry.pack()
        entries["x0"] = x0_entry

        Label(popup, text="Initial y (y0):").pack()
        y0_entry = Entry(popup, width=15)
        y0_entry.pack()
        entries["y0"] = y0_entry

        Label(popup, text="End Value of x (x_end):").pack()
        x_end_entry = Entry(popup, width=15)
        x_end_entry.pack()
        entries["x_end"] = x_end_entry

        Label(popup, text="Step Size (h):").pack()
        h_entry = Entry(popup, width=15)
        h_entry.pack()
        entries["h"] = h_entry
        

    elif method_type == "matrix_inverse":
        # MATRIX INVERSION TAKES ONLY A SQUARE MATRIX
        Label(popup, text="Enter a square matrix A (row by row, space-separated values):").pack()
        matrix_entry = Text(popup, height=5, width=30)
        matrix_entry.pack()
        entries["matrix"] = matrix_entry

    elif method_type in ["gauss_seidel", "jacobi"]:
        Label(popup, text="Enter matrix A (row by row, space-separated values):").pack()
        matrix_entry = Text(popup, height=5, width=30)
        matrix_entry.pack()
        entries["matrix"] = matrix_entry

        Label(popup, text="Enter constant vector B (space-separated values):").pack()
        b_entry = Entry(popup, width=30)
        b_entry.pack()
        entries["b"] = b_entry

        Label(popup, text="Enter initial guess X0 (optional, space-separated values):").pack()
        x0_entry = Entry(popup, width=30)
        x0_entry.pack()
        entries["x0"] = x0_entry

        # Add optional tolerance input
        Label(popup, text="Tolerance (optional, default=1e-6):").pack()
        tol_entry = Entry(popup, width=15)
        tol_entry.pack()
        entries["tol"] = tol_entry

        # Add optional max iterations input
        Label(popup, text="Max Iterations (optional, default=100):").pack()
        max_iter_entry = Entry(popup, width=15)
        max_iter_entry.pack()
        entries["max_iter"] = max_iter_entry

    elif method_type in ["gauss_eli", "gauss_jordan"]:
        Label(popup, text="Enter the number of variables:").pack()
        var_count_entry = Entry(popup, width=10)
        var_count_entry.pack()
        entries["var_count"] = var_count_entry

        Label(popup, text="Enter each equation’s coefficients (space-separated):").pack()
        matrix_entry = Text(popup, height=5, width=30)
        matrix_entry.pack()
        entries["matrix"] = matrix_entry

        Label(popup, text="Enter the constants vector B (space-separated values):").pack()
        b_entry = Entry(popup, width=30)
        b_entry.pack()
        entries["b"] = b_entry

    
    elif method_type == "lu":
        # LU FACTORIZATION REQUIRES A SQUARE MATRIX & CONSTANT VECTOR
        Label(popup, text="Enter a square matrix A (row by row, space-separated values):").pack()
        matrix_entry = Text(popup, height=5, width=30)
        matrix_entry.pack()
        entries["matrix"] = matrix_entry

        Label(popup, text="Enter the constant vector b (space-separated values):").pack()
        b_entry = Entry(popup, width=30)
        b_entry.pack()
        entries["b"] = b_entry
        
    elif method_type == "interpolation":
        Label(popup, text="Enter x values (comma-separated):").pack()
        x_entry = Entry(popup, width=30)
        x_entry.pack()
        entries["x_vals"] = x_entry

        Label(popup, text="Enter y values (comma-separated):").pack()
        y_entry = Entry(popup, width=30)
        y_entry.pack()
        entries["y_vals"] = y_entry

        Label(popup, text="Enter the x value to interpolate at:").pack()
        x_interp_entry = Entry(popup, width=15)
        x_interp_entry.pack()
        entries["x"] = x_interp_entry

    elif method_type == "integration":
        Label(popup, text="Enter function (in terms of x, e.g., x**2):").pack()
        func_entry = Entry(popup, width=30)
        func_entry.pack()
        entries["func"] = func_entry

        Label(popup, text="Enter lower limit (a):").pack()
        a_entry = Entry(popup, width=15)
        a_entry.pack()
        entries["a"] = a_entry

        Label(popup, text="Enter upper limit (b):").pack()
        b_entry = Entry(popup, width=15)
        b_entry.pack()
        entries["b"] = b_entry

        Label(popup, text="Enter number of intervals (n, must be even):").pack()
        n_entry = Entry(popup, width=15)
        n_entry.pack()
        entries["n"] = n_entry

    elif method_type == "differentiation":
        Label(popup, text="Enter function (in terms of x, e.g., x**3):").pack()
        func_entry = Entry(popup, width=30)
        func_entry.pack()
        entries["func"] = func_entry

        Label(popup, text="Enter x value for derivative:").pack()
        x_entry = Entry(popup, width=15)
        x_entry.pack()
        entries["x"] = x_entry

    
    elif method_name == "Simpson's 3/8 Rule":
        Label(popup, text="Enter function (in terms of x, e.g., x**2):").pack()
        func_entry = Entry(popup, width=30)
        func_entry.pack()
        entries["func"] = func_entry

        Label(popup, text="Enter lower limit (a):").pack()
        a_entry = Entry(popup, width=15)
        a_entry.pack()
        entries["a"] = a_entry

        Label(popup, text="Enter upper limit (b):").pack()
        b_entry = Entry(popup, width=15)
        b_entry.pack()
        entries["b"] = b_entry

        Label(popup, text="Enter number of intervals (n, must be a multiple of 3):").pack()
        n_entry = Entry(popup, width=15)
        n_entry.pack()
        entries["n"] = n_entry

    elif method_name == "Newton Backward Interpolation":
        Label(popup, text="Enter x values (comma-separated):").pack()
        x_entry = Entry(popup, width=30)
        x_entry.pack()
        entries["x_vals"] = x_entry

        Label(popup, text="Enter y values (comma-separated):").pack()
        y_entry = Entry(popup, width=30)
        y_entry.pack()
        entries["y_vals"] = y_entry

        Label(popup, text="Enter the x value to interpolate at:").pack()
        x_interp_entry = Entry(popup, width=15)
        x_interp_entry.pack()
        entries["x"] = x_interp_entry


    # Function to Show Code in a New Window
    def show_code():
        try:
            # Match method name with its respective file
            method_files = {
                "Bisection Method": "bisection_method.py",
                "False Position Method": "false_position_method.py",
                "Newton-Raphson Method": "newton_raphson_method.py",
                "Secant Method": "secant_method.py",
                "Runge-Kutta Method": "runge_kutta.py",
                "Gauss Elimination": "gauss_eli.py",
                "Gauss-Jordan": "gauss_jordan.py",
                "Gauss-Seidel Method": "gauss_seidel_iterative_method.py",
                "Jacobi Iterative Method": "jacobi_iterative_method.py",
                "LU Factorization": "lu_factorization.py",
                "Matrix Inversion": "matrix_inverse.py",
                
                # ✅ Integration methods (Both Simpson’s rules in one file)
                "Simpson's 1/3 Rule": "integration_methods.py",
                "Simpson's 3/8 Rule": "integration_methods.py",

                # ✅ Interpolation methods (Both Forward & Backward in one file)
                "Newton Forward Interpolation": "interpolation_methods.py",
                "Newton Backward Interpolation": "interpolation_methods.py",

                # ✅ Differentiation method (Second Order Derivative)
                "Second Order Derivative": "differentiation_methods.py"
                }

            filename = method_files.get(method_name)
            if not filename:
                messagebox.showerror("Error", "No file found for this method.")
                return

            filepath = os.path.join("src", filename)  # Adjust path if necessary

            with open(filepath, "r") as file:
                code = file.read()

            # Create a new window to show code
            code_window = Toplevel(popup)
            code_window.title(f"Code for {method_name}")
            code_window.geometry("600x500")

            text_widget = Text(code_window, wrap="none", font=("Courier", 10))
            text_widget.insert("1.0", code)
            text_widget.config(state=DISABLED)
            text_widget.pack(expand=True, fill="both")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load code: {e}")
            
            
    

    def run_method():
        try:
            result = None  # Ensure 'result' always exists

            # ✅ Matrix-Based Methods (Already included)
            # ✅ Matrix-Based Methods Handling
            if method_type == "matrix_inverse":
                try:
                    # Get matrix input from GUI
                    matrix_text = entries["matrix"].get("1.0", END).strip()
                    matrix = np.array([list(map(float, row.split())) for row in matrix_text.split("\n") if row.strip()])

                    # Ensure matrix is square
                    if matrix.shape[0] != matrix.shape[1]:
                        messagebox.showerror("Error", "Matrix must be square for inversion.")
                        return

                    # Compute matrix inverse
                    result = method_func(matrix)
                    
                    # ✅ Debugging: Check raw output
                    print("Raw result from method_func:", result)

                    # Ensure it's a NumPy array before conversion
                    if not isinstance(result, np.ndarray):
                        messagebox.showerror("Error", "Invalid computation: Expected a NumPy array but got something else.")
                        return

                    # ✅ Convert NumPy array to Python list before formatting
                    result_list = result.tolist()
                    print("Converted list result:", result_list)  # Debugging step

                    # ✅ Format matrix nicely for better alignment
                    formatted_result = "\n".join([" | ".join(f"{val:10.6f}" for val in row) for row in result_list])

                    # ✅ Display formatted result with a title and line separation
                    messagebox.showinfo("Matrix Inversion Result", f"Inverted Matrix:\n\n" + formatted_result)


                except ValueError:
                    messagebox.showerror("Input Error", "Invalid matrix input. Ensure numbers are formatted correctly.")
                except np.linalg.LinAlgError:
                    messagebox.showerror("Error", "Matrix is singular and cannot be inverted.")
                except Exception as e:
                    messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {e}")



            elif method_type == "lu":
                try:
                    matrix_text = entries["matrix"].get("1.0", END).strip()
                    b_text = entries["b"].get().strip()

                    matrix = np.array([list(map(float, row.split())) for row in matrix_text.split("\n") if row.strip()])
                    b = np.array(list(map(float, b_text.split())))

                    if matrix.shape[0] != matrix.shape[1]:
                        messagebox.showerror("Error", "Matrix must be square for LU Factorization.")
                        return

                    if matrix.shape[0] != len(b):
                        messagebox.showerror("Error", "Matrix A row count must match the size of vector B.")
                        return

                    # Perform LU decomposition
                    L, U, y, x = lu_factorization.lu_factorization(matrix, b)

                    # ✅ Improved formatted result display
                    result = (
                        f"Lower Matrix L:\n{L}\n\n"
                        f"Upper Matrix U:\n{U}\n\n"
                        f"Intermediate y:\n{y}\n\n"
                        f"Solution x:\n{x}"
                    )

                    # messagebox.showinfo("LU Factorization Result", result)

                except ValueError:
                    messagebox.showerror("Input Error", "Invalid matrix/vector input. Ensure correct number formatting.")
                    return


            elif method_type in ["gauss_seidel", "jacobi"]:
                try:
                    matrix_text = entries["matrix"].get("1.0", END).strip()
                    matrix = np.array([list(map(float, row.split())) for row in matrix_text.split("\n") if row.strip()])

                    b = np.array(list(map(float, entries["b"].get().split())))
                    x0 = np.zeros(len(b)) if not entries["x0"].get() else np.array(list(map(float, entries["x0"].get().split())))

                    # ✅ Ensure correct matrix dimensions
                    if matrix.shape[0] != len(b):
                        messagebox.showerror("Error", "Matrix A row count must match the size of vector B.")
                        return

                    # Get tolerance value or set default
                    tol = float(entries["tol"].get()) if entries["tol"].get() else 1e-6

                    # Get max iterations or set default
                    max_iter = int(entries["max_iter"].get()) if entries["max_iter"].get() else 100

                    result = method_func(matrix, b, x0, tol, max_iter)

                except ValueError:
                    messagebox.showerror("Input Error", "Invalid matrix/vector input. Ensure correct number formatting.")
                    return

        


            elif method_type in ["gauss_eli", "gauss_jordan"]:
                try:
                    var_count = int(entries["var_count"].get())
                    matrix_text = entries["matrix"].get("1.0", END).strip()
                    b_text = entries["b"].get().strip()

                    matrix = np.array([list(map(float, row.split())) for row in matrix_text.split("\n") if row.strip()])
                    b = np.array(list(map(float, b_text.split())))

                    # ✅ Validate size constraints
                    if matrix.shape[0] != var_count or len(b) != var_count:
                        messagebox.showerror("Error", "Matrix rows and vector size must match the number of variables.")
                        return

                    result = method_func(matrix, b)

                except ValueError:
                    messagebox.showerror("Input Error", "Invalid matrix/vector input. Ensure correct number formatting.")
                    return

            elif method_type == "interpolation":
                try:
                    x_vals = list(map(float, entries["x_vals"].get().split(",")))
                    y_vals = list(map(float, entries["y_vals"].get().split(",")))
                    x_interp = float(entries["x"].get())

                    # ✅ Ensure X & Y have the same number of values
                    if len(x_vals) != len(y_vals):
                        messagebox.showerror("Input Error", "X and Y values must have the same count.")
                        return

                    # ✅ Distinguish between Forward and Backward Interpolation
                    if method_name == "Newton Forward Interpolation":
                        result = newton_forward_interpolation(x_vals, y_vals, x_interp)
                    elif method_name == "Newton Backward Interpolation":
                        result = newton_backward_interpolation(x_vals, y_vals, x_interp)

                except ValueError:
                    messagebox.showerror("Input Error", "Invalid format! Use only numbers and separate values with commas.")
                    return

            # ✅ Handle Numerical Integration Methods
            elif method_type == "integration":
                try:
                    x = sp.Symbol('x')
                    func_expr = sp.sympify(entries["func"].get())  # Convert input function
                    f = sp.lambdify(x, func_expr, "math")

                    a = float(entries["a"].get())
                    b = float(entries["b"].get())
                    n = int(entries["n"].get()) if entries["n"].get() else 100

                    # ✅ Ensure correct interval conditions for Simpson’s Rules
                    if method_name == "Simpson's 1/3 Rule" and n % 2 == 1:
                        messagebox.showerror("Input Error", "Number of intervals (n) must be even for Simpson's 1/3 Rule.")
                        return

                    if method_name == "Simpson's 3/8 Rule" and n % 3 != 0:
                        messagebox.showerror("Input Error", "Number of intervals (n) must be a multiple of 3 for Simpson's 3/8 Rule.")
                        return

                    # ✅ Ensure `a < b`
                    if a >= b:
                        messagebox.showerror("Input Error", "Lower limit (a) must be less than upper limit (b).")
                        return

                    result = method_func(f, a, b, n)

                except ValueError:
                    messagebox.showerror("Input Error", "Invalid input! Please enter valid numbers.")
                    return


            # ✅ Handle Numerical Differentiation Methods
            elif method_type == "differentiation":
                try:
                    x = sp.Symbol('x')
                    func_expr = sp.sympify(entries["func"].get())  # Convert input function
                    f = sp.lambdify(x, func_expr, "math")  # Convert to numerical function

                    x_value = float(entries["x"].get())

                    # Compute the second-order derivative
                    result = method_func(f, x_value)

                    # Display result
                    # messagebox.showinfo("Second Order Derivative", f"f''({x_value}) = {result:.6f}")

                except ValueError:
                    messagebox.showerror("Input Error", "Invalid input! Please enter a valid function and numeric x value.")
                except Exception as e:
                    messagebox.showerror("Error", f"Invalid input: {e}")



            # ✅ Fix for Root-Finding & Runge-Kutta Methods
            elif method_type in ["root_finding", "newton_raphson", "secant", "runge_kutta"]:
                func_expr = sp.sympify(entries["func"].get())  # Convert function input

                # Convert function into numerical function using sp.lambdify()
                f_numeric = sp.lambdify(["x", "y"], func_expr, "math")

                if method_type == "root_finding":
                    try:
                        a = float(entries["a"].get())
                        b = float(entries["b"].get())

                        # ✅ Ensure valid range for root-finding methods
                        if a >= b:
                            messagebox.showerror("Input Error", "Lower bound (a) must be less than upper bound (b)")
                            return

                        # Get tolerance value or set default
                        tol = float(entries["tol"].get()) if entries["tol"].get() else 1e-6

                        # Get max iterations or set default
                        max_iter = int(entries["max_iter"].get()) if entries["max_iter"].get() else 100

                        result = method_func(func_expr, a, b, tol, max_iter)

                    except ValueError:
                        messagebox.showerror("Input Error", "Invalid numeric input. Please enter valid numbers.")
                        return


                elif method_type == "newton_raphson":
                    try:
                        x0 = float(entries["x0"].get())

                        # Get tolerance value or set default
                        tol = float(entries["tol"].get()) if entries["tol"].get() else 1e-6

                        # Get max iterations or set default
                        max_iter = int(entries["max_iter"].get()) if entries["max_iter"].get() else 100

                        result = method_func(func_expr, x0, tol, max_iter)

                    except ValueError:
                        messagebox.showerror("Input Error", "Invalid numeric input. Please enter a valid number for x0.")
                        return

                
                elif method_type == "secant":
                    try:
                        x0 = float(entries["x0"].get())
                        x1 = float(entries["x1"].get())

                        # ✅ Ensure x0 and x1 are distinct
                        if x0 == x1:
                            messagebox.showerror("Input Error", "x0 and x1 must be different for the Secant method.")
                            return

                        # Get tolerance value or set default
                        tol = float(entries["tol"].get()) if entries["tol"].get() else 1e-6

                        # Get max iterations or set default
                        max_iter = int(entries["max_iter"].get()) if entries["max_iter"].get() else 100

                        result = method_func(func_expr, x0, x1, tol, max_iter)

                    except ValueError:
                        messagebox.showerror("Input Error", "Invalid numeric input. Please enter valid numbers for x0 and x1.")
                        return



                elif method_type == "runge_kutta":
                    func_expr = sp.sympify(entries["func"].get())  # Convert input function
                    f_lambda = sp.lambdify(["x", "y"], func_expr, "math")  # Convert to numerical function

                    x0 = float(entries["x0"].get())
                    y0 = float(entries["y0"].get())
                    x_end = float(entries["x_end"].get())
                    h = float(entries["h"].get())

                    # Call the Runge-Kutta method
                    result = method_func(f_lambda, x0, y0, x_end, h)

                    # Format output neatly
                    formatted_result = "Runge-Kutta Method Solution:\n"
                    formatted_result += "-" * 40 + "\n"
                    formatted_result += "{:<10} {:<15}\n".format("x", "y(x)")
                    formatted_result += "-" * 40 + "\n"
                    
                    for x, y in result:
                        formatted_result += "{:<10.4f} {:<15.6f}\n".format(x, y)

                    # Show results in a pop-up message box
                    messagebox.showinfo("Runge-Kutta Results", formatted_result)
                    return


            if result is not None:
                if isinstance(result, tuple):  # ✅ Only root-finding methods return a tuple (root, iterations)
                    root, iterations = result
                    formatted_iterations = "\n".join([f"Iteration {i+1}: {val:.6f}" for i, val in enumerate(iterations)])
                    message = f"Final Root: {root:.6f}\n\nIterations:\n{formatted_iterations}"

                elif isinstance(result, np.ndarray):  # ✅ Jacobi returns a solution vector (array)
                    formatted_solution = "\n".join([f"x{i+1} = {val:.6f}" for i, val in enumerate(result)])
                    message = f"Solution:\n{formatted_solution}"

                else:  # ✅ Handling for other numerical methods
                    message = f"Solution: {result}"

                # ✅ Show result correctly
                messagebox.showinfo("Result", message)

                # ✅ Save only relevant data
                with open("results.txt", "a") as file:
                    file.write(f"{method_name} -> Solution:\n")
                    if isinstance(result, np.ndarray):
                        file.write("\n".join([f"x{i+1} = {val:.6f}" for i, val in enumerate(result)]))
                    else:
                        file.write(f"{result}")
                    file.write("\n\n")



            else:
                messagebox.showerror("Error", "No valid result was computed.")

        except ValueError as ve:
            messagebox.showerror("Error", f"Value Error: {ve}")

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")


    Button(popup, text="Run", command=run_method).pack(pady=10)
    Button(popup, text="Show Code", command=show_code).pack(pady=5)

# Bind methods to buttons with correct parameters

def show_previous_results():
        try:
            with open("results.txt", "r") as file:
                results = file.readlines()[-5:]  # Show last 5 results

            if results:
                messagebox.showinfo("Previous Results", "".join(results))
            else:
                messagebox.showinfo("Previous Results", "No previous results found.")

        except FileNotFoundError:
            messagebox.showinfo("Previous Results", "No previous results found.")
            

def visualize_root_finding(method_name, entries):
    """Visualizes the function and root-finding process."""
    try:
        # Get function input from user
        x = sp.Symbol('x')
        func_expr = sp.sympify(entries["func"].get())  # Convert input function
        f = sp.lambdify(x, func_expr, "numpy")  # Convert function to numpy function

        # Initialize iteration history
        iter_history = []


        # Determine the root-finding method
        if method_name in ["Bisection Method", "False Position Method"]:
            a = float(entries["a"].get())
            b = float(entries["b"].get())
            x_vals = np.linspace(a - 1, b + 1, 400)  # Generate x values for plotting

            if method_name == "Bisection Method":
                from bisection_method import bisection_method
                root, iter_history = bisection_method(func_expr, a, b, tol=1e-6, max_iter=100)
            else:
                from false_position_method import false_position_method
                root, iter_history = false_position_method(func_expr, a, b, tol=1e-6, max_iter=100)

        elif method_name == "Newton-Raphson Method":
            x0 = float(entries["x0"].get())
            from newton_raphson_method import newton_raphson_method
            root, iter_history = newton_raphson_method(func_expr, x0, tol=1e-6, max_iter=100)
            x_vals = np.linspace(x0 - 2, x0 + 2, 400)  # Generate x values for plotting

        elif method_name == "Secant Method":
            x0 = float(entries["x0"].get())
            x1 = float(entries["x1"].get())
            from secant_method import secant_method
            root, iter_history = secant_method(func_expr, x0, x1, tol=1e-6, max_iter=100)
            x_vals = np.linspace(x0 - 2, x1 + 2, 400)  # Generate x values for plotting

        else:
            messagebox.showerror("Error", "Visualization is not available for this method.")
            return

        # Compute y values
        y_vals = f(x_vals)

        # Plot function
        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, y_vals, label=f"f(x) = {func_expr}", color="blue")
        plt.axhline(0, color="black", linewidth=1)  # x-axis
        plt.axvline(root, color="red", linestyle="--", label=f"Root ≈ {root:.6f}")

        # Plot iteration points
        # Plot all iteration points in green
        plt.scatter(iter_history[:-1], [f(val) for val in iter_history[:-1]], color="green", label="Iterations", zorder=3)

        # Highlight the final iteration in red
        plt.scatter(iter_history[-1], f(iter_history[-1]), color="purple", label="Final Iteration", zorder=3, marker="o", edgecolors="black", s=100)


        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"{method_name} - Root Approximation")
        plt.legend()
        plt.grid()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Graph could not be generated: {e}")

# ==== SECTION: Solution of Linear Equations ====
linear_frame = LabelFrame(main_frame, text="Solution of Linear Equations", font=("Arial", 12, "bold"), bg="#1E1E1E", fg="white")
linear_frame.grid(row=0, column=0, padx=20, pady=10, sticky="w")

Button(linear_frame, text="Jacobi Iterative Method", command=lambda: get_input_and_run("Jacobi Iterative Method", jacobi_iterative_method.jacobi_iterative_method, "jacobi"), **button_style).pack(pady=3)
Button(linear_frame, text="Gauss-Seidel Iterative Method", command=lambda: get_input_and_run("Gauss-Seidel Method", gauss_seidel_iterative_method.gauss_seidel_iterative_method, "gauss_seidel"), **button_style).pack(pady=3)
Button(linear_frame, text="Gauss Elimination", command=lambda: get_input_and_run("Gauss Elimination", gauss_eli.gauss_eli, "gauss_eli"), **button_style).pack(pady=3)
Button(linear_frame, text="Gauss-Jordan Elimination", command=lambda: get_input_and_run("Gauss-Jordan", gauss_jordan.gauss_jordan, "gauss_jordan"), **button_style).pack(pady=3)
Button(linear_frame, text="LU Factorization", command=lambda: get_input_and_run("LU Factorization", lu_factorization.lu_factorization, "lu"), **button_style).pack(pady=3)

# ==== SECTION: Solution of Non-Linear Equations ====
nonlinear_frame = LabelFrame(main_frame, text="Solution of Non-Linear Equations", font=("Arial", 12, "bold"), bg="#1E1E1E", fg="white")
nonlinear_frame.grid(row=0, column=1, padx=20, pady=10, sticky="w")

Button(nonlinear_frame, text="Bisection Method", command=lambda: get_input_and_run("Bisection Method", bisection_method.bisection_method, "root_finding"), **button_style).pack(pady=3)
Button(nonlinear_frame, text="False Position Method", command=lambda: get_input_and_run("False Position Method", false_position_method.false_position_method, "root_finding"), **button_style).pack(pady=3)
Button(nonlinear_frame, text="Secant Method", command=lambda: get_input_and_run("Secant Method", secant_method.secant_method, "secant"), **button_style).pack(pady=3)
Button(nonlinear_frame, text="Newton-Raphson Method", command=lambda: get_input_and_run("Newton-Raphson Method", newton_raphson_method.newton_raphson_method, "newton_raphson"), **button_style).pack(pady=3)

# ==== SECTION: Solution of Differential Equations ====
differential_frame = LabelFrame(main_frame, text="Solution of Differential Equations", font=("Arial", 12, "bold"), bg="#1E1E1E", fg="white")
differential_frame.grid(row=1, column=0, padx=20, pady=10, sticky="w")

Button(differential_frame, text="Runge-Kutta Method", command=lambda: get_input_and_run("Runge-Kutta Method", runge_kutta.runge_kutta_method, "runge_kutta"), **button_style).pack(pady=3)

# ==== SECTION: Matrix Inversion ====
matrix_frame = LabelFrame(main_frame, text="Matrix Inversion", font=("Arial", 12, "bold"), bg="#1E1E1E", fg="white")
matrix_frame.grid(row=1, column=1, padx=20, pady=10, sticky="w")

Button(matrix_frame, text="Matrix Inversion", command=lambda: get_input_and_run("Matrix Inversion", matrix_inverse.matrix_inverse, "matrix_inverse"), **button_style).pack(pady=3)

# ==== SECTION: Interpolation, Integration, and Differentiation ====
calculus_frame = LabelFrame(main_frame, text="Interpolation, Integration, and Differentiation", font=("Arial", 12, "bold"), bg="#1E1E1E", fg="white")
calculus_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="w")

Button(calculus_frame, text="Newton Forward Interpolation", command=lambda: get_input_and_run("Newton Forward Interpolation", newton_forward_interpolation, "interpolation"), **button_style).pack(pady=3)
Button(calculus_frame, text="Newton Backward Interpolation", command=lambda: get_input_and_run("Newton Backward Interpolation", newton_backward_interpolation, "interpolation"), **button_style).pack(pady=3)
Button(calculus_frame, text="Simpson's 1/3 Rule", command=lambda: get_input_and_run("Simpson's 1/3 Rule", simpsons_one_third_rule, "integration"), **button_style).pack(pady=3)
Button(calculus_frame, text="Simpson's 3/8 Rule", command=lambda: get_input_and_run("Simpson's 3/8 Rule", simpsons_three_eighth_rule, "integration"), **button_style).pack(pady=3)
Button(calculus_frame, text="Second Order Derivative", command=lambda: get_input_and_run("Second Order Derivative", second_order_derivative, "differentiation"), **button_style).pack(pady=3)

# ==== Button to View Previous Results ====
Button(root, text="View Previous Results", command=show_previous_results, width=30, font=("Arial", 10), bg="green", fg="white").pack(pady=10)
Button(root, text="Help", command=open_help_window, width=30, font=("Arial", 10), bg="orange", fg="black").pack(pady=10)


# ==== Footer Section ====
Label(root, text="Developed by Adit Mugdha Das and © 2025 All rights reserved", font=("Arial", 10, "bold"), bg="#1E1E1E", fg="white").pack(side=BOTTOM, pady=10)



root.mainloop()


