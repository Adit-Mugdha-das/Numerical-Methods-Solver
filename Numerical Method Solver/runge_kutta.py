import math

def runge_kutta_method(f, x0, y0, x_end, h):
    """Solves dy/dx using the Runge-Kutta Method and returns the computed values."""
    x = x0
    y = y0
    results = [(x, y)]  # Store results in a list

    while x < x_end:
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(x + h, y + k3)
        y = y + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x = round(x + h, 6)  # Prevent floating-point precision errors
        results.append((x, y))

    return results  # Return the computed values instead of printing


if __name__ == "__main__":
    try:
        def f(x, y):
            return eval(func, {"x": x, "y": y, "math": math})

        print("\nNote: Use 'math' functions for complex expressions (e.g., 'math.sin(x)', 'math.exp(x)', etc.)")
        func = input("Enter the function f(x, y) for dy/dx: ")
        x0 = float(input("Enter the initial value of x (x0): "))
        y0 = float(input("Enter the initial value of y (y0): "))
        x_end = float(input("Enter the value of x at which to find y (x_end): "))
        h = float(input("Enter the step size (h): "))

        results = runge_kutta_method(f, x0, y0, x_end, h)
        
        # Print results
        print("\nRunge-Kutta Method Results:")
        for x, y in results:
            print(f"x = {x:.4f}, y = {y:.4f}")

    except Exception as e:
        print(f"Error: {e}")
