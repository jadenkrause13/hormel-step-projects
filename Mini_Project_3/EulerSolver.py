#THIS CODE EVALUATES EQUATIONS NUMERICALLY USING EULER'S METHOD

import numpy as np
import matplotlib.pyplot as plt

def euler_solver(f_expr, t0, tf, y0, h):
    """
    Solves an ODE dy/dt = f(t, y) using the Explicit Euler Method.
    Evaluates the equation dynamically from a string expression.
    """
    # Create the time grid
    t_values = np.arange(t0, tf + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    
    # Pre-compile the mathematical expression for efficiency and safety
    code = compile(f_expr, '<string>', 'eval')

    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]
        
        # Evaluate the user's expression given local variables 't' and 'y'
        # Also includes 'np' so the user can use math tools like np.sin()
        f_val = eval(code, {"np": np, "t": t, "y": y, "math": np})
        
        y_values[i] = y + h * f_val

    return t_values, y_values

if __name__ == "__main__":
    print("=== Interactive Euler ODE Solver ===")
    print("You can use standard operations: +, -, *, /, ** (for exponentiation)")
    print("You can also use NumPy functions like: np.sin(t), np.cos(t), np.exp(-t)\n")
    
    # 1. Ask user for inputs
    equation = input("Enter the right-hand side of dy/dt = f(t, y): ")
    t0 = float(input("Enter initial time t0 = "))
    tf = float(input("Enter final time tf = "))
    y0 = float(input(f"Enter initial condition y({t0:.0f}) = "))
    h = float(input("Enter step size h = "))
    
    try:
        # 2. Run the solver
        t_res, y_res = euler_solver(equation, t0, tf, y0, h)
        
        # 3. Print a quick summary of the results
        print("\n=== First 10 Computed Steps ===")
        for i in range(min(10, len(t_res))):
            print(f"Step {i}: t = {t_res[i]:.3f}, y = {y_res[i]:.4f}")
            
        print(f"\nFinal Result at t = {t_res[-1]:.3f}: y = {y_res[-1]:.4f}")

        # 4. Display the solution curve
        plt.figure(figsize=(8, 4))
        plt.plot(t_res, y_res, 'bo-', markersize=4, label=f"Euler Approx: dy/dt = {equation}")
        plt.title('Euler Method Solution')
        plt.xlabel('t')
        plt.ylabel('y(t)')
        plt.grid(True)
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"\nError: Could not evaluate the expression. Details: {e}")
