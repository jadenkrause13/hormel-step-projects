import numpy as np
import matplotlib.pyplot as plt

def rk4_step(y, v, b, k, m, dt):
    """Calculates the next state using 4th-Order Runge-Kutta integration."""
    # First derivative equations: dy/dt = v, dv/dt = -(b*v + k*y)/m
    f_y = lambda y_val, v_val: v_val
    f_v = lambda y_val, v_val: -(b * v_val + k * y_val) / m

    # k1 slopes
    k1_y = f_y(y, v)
    k1_v = f_v(y, v)

    # k2 slopes (midpoint)
    k2_y = f_y(y + 0.5 * dt * k1_y, v + 0.5 * dt * k1_v)
    k2_v = f_v(y + 0.5 * dt * k1_y, v + 0.5 * dt * k1_v)

    # k3 slopes (midpoint)
    k3_y = f_y(y + 0.5 * dt * k2_y, v + 0.5 * dt * k2_v)
    k3_v = f_v(y + 0.5 * dt * k2_y, v + 0.5 * dt * k2_v)

    # k4 slopes (endpoint)
    k4_y = f_y(y + dt * k3_y, v + dt * k3_v)
    k4_v = f_v(y + dt * k3_y, v + dt * k3_v)

    # Weighted update
    next_y = y + (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
    next_v = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    
    return next_y, next_v

def run_custom_solver():
    print("--- Custom Mass-Spring-Damper Numerical Solver ---")
    
    # 1. Accept dynamic terminal inputs from user
    try:
        m = float(input("Enter mass (m) measured in kg [e.g., 1]: ") or 1.0)
        b = float(input("Enter damping coefficient (b) measured in kg/s [e.g., 6]: ") or 6.0)
        k = float(input("Enter spring constant (k) measured in N/m [e.g., 16]: ") or 16.0)
        y0 = float(input("Enter initial position y(0) measured in m: ") or 1.0)
        v0 = float(input("Enter initial velocity y'(0) measured in m/s: ") or 0.0)
        t_max = float(input("Enter simulation runtime duration (seconds): ") or 5.0)
    except ValueError:
        print("Invalid numerical input. Running with default system parameters.")
        m, b, k, y0, v0, t_max = 1.0, 6.0, 16.0, 1.0, 0.0, 5.0

        #NOTES:
        #damping coefficient is also measured in N*s/m, and is fundamentally a measure of external 
        #forces on the system such as friction and air resistance. 
        #initial position is distance measured from the spring's equilibrium, and as such can be 
        #positive or negative.
        #similarly, initial velocity can also be measured as positive or negative, as speed going 
        #out or coming in.

    # 2. Setup simulation domain arrays
    dt = 0.005  # Fixed time step size
    num_steps = int(t_max / dt) + 1
    t_array = np.linspace(0, t_max, num_steps)
    y_array = np.zeros(num_steps)
    v_array = np.zeros(num_steps)

    # 3. Apply state configurations
    y_array[0] = y0
    v_array[0] = v0

    # 4. Step through time vector using the RK4 engine
    for i in range(0, num_steps - 1):
        y_array[i+1], v_array[i+1] = rk4_step(y_array[i], v_array[i], b, k, m, dt)

    # 5. Determine system damping type classification label
    discriminant = b**2 - 4*m*k
    if b == 0:
        system_type = "Undamped"
    elif discriminant < 0:
        system_type = "Underdamped"
    elif discriminant == 0:
        system_type = "Critically Damped"
    else:
        system_type = "Overdamped"

    print(f"System Type: {system_type}")

    # 6. Generate Clean Visualization Profile
    plt.figure(figsize=(10, 5))
    plt.plot(t_array, y_array, label=f"Position y(t) ({system_type})", color="#1f77b4", linewidth=2)
    plt.plot(t_array, v_array, label="Velocity y'(t)", color="#ff7f0e", linestyle="--", linewidth=1.5)
    
    plt.title(f"Numerical Trajectory Simulation (m={m}, b={b}, k={k})")
    plt.xlabel("Time (t)")
    plt.ylabel("System State Amplitude")
    plt.axhline(0, color="black", linewidth=0.8, linestyle=":")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# Run the live interactive terminal routine
if __name__ == "__main__":
    run_custom_solver()
