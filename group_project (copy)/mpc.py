import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from cubic_spline_trajectory import generate_cubic_spline
from model import export_kinematic_bicycle_model  # Import updated model
def setup_mpc_controller():
    """Setup the MPC controller using acados."""

    # Define parameters
    NX = 4  # State dimensions: [x, y, v, theta]
    NU = 2  # Control dimensions: [a, phi]
    T = 5.0  # Total time horizon [s]
    DT = 0.2  # Time step [s]
    N = int(T / DT)  # Prediction horizon

    Q = np.diag([10.0, 10.0, 1.0, 1.0])  # Stronger state penalties
    R = np.diag([0.1, 0.1])  # Slightly stronger control effort penalties
    Qf = np.diag([20.0, 20.0, 2.0, 2.0])  # Terminal penalties

    # Vehicle constraints
    MAX_STEER = np.deg2rad(45.0)  # Max steering angle [rad]
    MAX_ACCEL = 1.0  # Max acceleration [m/s^2]
    MAX_SPEED = 55.0 / 3.6  # Max speed [m/s]

    # Create the model
    model = export_kinematic_bicycle_model()  # Use the updated model

    # Initialize the OCP
    ocp = AcadosOcp()
    ocp.model = model

    # Set dimensions
    ocp.dims.N = N  # Number of shooting intervals

    # Discretization
    ocp.solver_options.tf = T  # Total time horizon

    # Cost function
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.block([
        [Q, np.zeros((NX, NU))],
        [np.zeros((NU, NX)), R]
    ])
    ocp.cost.W_e = Qf

    # Cost references
    ocp.cost.yref = np.zeros(NX + NU)  # [x_ref, y_ref, v_ref, theta_ref, a_ref, phi_ref]
    ocp.cost.yref_e = np.zeros(NX)  # Terminal reference: [x_ref, y_ref, v_ref, theta_ref]

    # Constraints
    ocp.constraints.lbu = np.array([-MAX_ACCEL, -MAX_STEER])
    ocp.constraints.ubu = np.array([MAX_ACCEL, MAX_STEER])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-np.inf, -np.inf, 0.0, -np.inf])  # Minimum speed >= 0
    ocp.constraints.ubx = np.array([np.inf, np.inf, MAX_SPEED, np.inf])
    ocp.constraints.x0 = np.zeros(NX)

    # Solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = "ERK"

    return ocp, model, N

def plot_results(waypoints, smooth_x, smooth_y, trajectory):
    """Plot the cubic spline path, waypoints, and MPC optimized trajectory."""
    plt.figure(figsize=(8, 6))
    plt.plot(smooth_x, smooth_y, label="Cubic Spline Path", linestyle="--")
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="MPC Optimized Path", color="orange")
    plt.scatter(trajectory[:, 0], trajectory[:, 1], label="MPC Points", color="blue", s=10)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Comparison of Cubic Spline Path and MPC Optimized Path")
    plt.legend()
    plt.grid()
    plt.show()

def run_mpc():
    """Run the MPC controller."""

    # Load waypoints from cubic spline
    waypoints = [
        [-6.5, 2.5],
        [-4.5, 2],
        [-3.5, 1],
        [-1.5, 0],
        [1, -1.5],
        [3, -2],
        [4.5, -2.5],
        [6, -3.5]
    ]
    waypoints = np.array(waypoints)

    # Generate smooth cubic spline path
    smooth_x, smooth_y = generate_cubic_spline(waypoints)

    # Initialize MPC
    ocp, model, N = setup_mpc_controller()
    solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    # Set initial state
    initial_state = np.array([smooth_x[0], smooth_y[0], 0.0, 0.0])  # [x, y, v, theta]
    solver.set(0, "x", initial_state)

    # Define reference trajectory
    for i in range(N):
        if i < len(smooth_x):
            x_ref = smooth_x[i]
            y_ref = smooth_y[i]
        else:
            x_ref = smooth_x[-1]
            y_ref = smooth_y[-1]

        y_ref = np.array([x_ref, y_ref, 0.0, 0.0, 0.0, 0.0])  # [x, y, v, theta, a, phi]
        solver.set(i, "yref", y_ref)

    solver.set(N, "yref_e", np.array([smooth_x[-1], smooth_y[-1], 0.0, 0.0]))

    # Solve MPC
    status = solver.solve()
    if status != 0:
        print("Solver failed with status:", status)
        return

    # Extract trajectory
    trajectory = np.array([solver.get(i, "x") for i in range(N + 1)])
    print("Optimized trajectory:", trajectory)

    # Plot results
    plot_results(waypoints, smooth_x, smooth_y, trajectory)

if __name__ == "__main__":
    run_mpc()