#THIS CODE SOLVES FOR THE NORMAL FREQUENCIES OF A COUPLED MASS-SPRING SYSTEM. IT IS CONSTRUED TO FOSTER CONVENIENT MANIPULATION AND CHANGING OF VALUES.
import sympy as sp

# Initialize terminal-friendly pretty printing for SymPy symbols
sp.init_printing(use_unicode=True)

# Define baseline hardcoded defaults
m1, m2 = 1.0, 1.0
k1, k2, k3 = 1.0, 1.0, 1.0
x1_0, v1_0 = 0.0, 0.0
x2_0, v2_0 = 2.0, 0.0

print("===============================================================")
print("             COUPLED MASS-SPRING SYSTEM SOLVER                 ")
print("===============================================================")

# Quick skip option
mode = input("Press ENTER to use all defaults, or type 'y' to customize parameters: ").strip().lower()

if mode == 'y':
    print("\n--- Custom Setup Wizard ---")
    print("Press ENTER at any prompt to keep its individual default value.\n")
    
    def get_validated_input(prompt, default_val, strictly_positive=False):
        """
        Prompts user for a float value, applies default on ENTER,
        and enforces numerical validation parameters.
        """
        while True:
            user_val = input(f"{prompt} [{default_val}]: ").strip()
            if not user_val:
                return float(default_val)
            
            try:
                parsed_val = float(user_val)
                
                # Check for physical boundaries (Masses and Springs must be > 0)
                if strictly_positive and parsed_val <= 0:
                    print("  ⚠️ Error: This value must be greater than 0. Please try again.")
                    continue
                    
                return parsed_val
                
            except ValueError:
                print("  ⚠️ Error: Invalid text entered. Please type a valid number.")

    # Collect custom parameters with physical validation rules
    m1   = get_validated_input("Enter mass m1", m1, strictly_positive=True)
    m2   = get_validated_input("Enter mass m2", m2, strictly_positive=True)
    k1   = get_validated_input("Enter spring k1", k1, strictly_positive=True)
    k2   = get_validated_input("Enter spring k2", k2, strictly_positive=True)
    k3   = get_validated_input("Enter spring k3", k3, strictly_positive=True)
    
    # Initial conditions can be negative, zero, or positive, so strictly_positive=False
    x1_0 = get_validated_input("Enter initial position x1(0)", x1_0, strictly_positive=False)
    v1_0 = get_validated_input("Enter initial velocity x1'(0)", v1_0, strictly_positive=False)
    x2_0 = get_validated_input("Enter initial position x2(0)", x2_0, strictly_positive=False)
    v2_0 = get_validated_input("Enter initial velocity x2'(0)", v2_0, strictly_positive=False)

print("\n===============================================================")
print("⚡ RUNNING SIMULATION WITH THE FOLLOWING VALUES:")
print(f"Masses:  m1 = {m1}, m2 = {m2}")
print(f"Springs: k1 = {k1}, k2 = {k2}, k3 = {k3}")
print(f"Initial: x1(0) = {x1_0}, x1'(0) = {v1_0} | x2(0) = {x2_0}, x2'(0) = {v2_0}")
print("===============================================================\n")

# 1. DEFINE GOVERNING SYSTEM MATRIX
A = sp.Matrix([
    [-(k1 + k2) / m1, k2 / m1],
    [k2 / m2, -(k2 + k3) / m2]
])

print("Governing System Matrix (A):")
sp.pprint(A)

# 2. CALCULATE NORMAL FREQUENCIES
print("\n--- Part 1: Normal Frequencies ---")
eigenvals = list(A.eigenvals().keys())

omega_1 = sp.sqrt(-eigenvals[0])
omega_2 = sp.sqrt(-eigenvals[1]) if len(eigenvals) > 1 else omega_1

print(f"omega_1 = {omega_1.evalf():.4f}")
print(f"omega_2 = {omega_2.evalf():.4f}")

# 3. SOLVE THE INITIAL VALUE PROBLEM
print("\n--- Part 2: Initial Value Problem Solution ---")
t = sp.Symbol('t', real=True)
x1 = sp.Function('x1')(t)
x2 = sp.Function('x2')(t)

# Map matrix operations into explicit differential equations
eq1 = sp.Eq(x1.diff(t, 2), A[0,0]*x1 + A[0,1]*x2)
eq2 = sp.Eq(x2.diff(t, 2), A[1,0]*x1 + A[1,1]*x2)

ics = {
    x1.subs(t, 0): x1_0,
    x1.diff(t).subs(t, 0): v1_0,
    x2.subs(t, 0): x2_0,
    x2.diff(t).subs(t, 0): v2_0
}

solutions = sp.dsolve([eq1, eq2], [x1, x2], ics=ics)

print("Exact solutions x1(t) and x2(t):")
for sol in solutions:
    sp.pprint(sol)
    print()
