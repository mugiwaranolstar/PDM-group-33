import matplotlib.pyplot as plt
import time
import cvxpy
import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import os
import pandas as pd
import random

###############################################################################
# 1) LOAD DATA & DECLARE GLOBAL CONSTANTS
###############################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
path_path = os.path.join(script_dir, "..", "csv_files", "best_path.csv")

best_path_array = pd.read_csv(path_path)
best_path_array = np.array(best_path_array)

# Obstacles from CSV (for visualization and constraints)
obstacles_path = os.path.join(script_dir, "..", "csv_files", "obstacles.csv")
obstacles_for_display = np.array(pd.read_csv(obstacles_path))

# Problem dimensions & horizon
NX = 4
NU = 2
T = 75
DT = 0.05

# MPC weighting matrices
R = np.diag([0.1, 0.001])         # Control input cost
Rd = np.diag([0.01, 0.1])         # Control input difference cost
Q = np.diag([1e-5, 1e-5, 0.1, 0.1])  # State deviation cost
Qf = 10 * Q                       # Final state deviation cost

print(f"Using T={T}, DT={DT}, total horizon = {T*DT}s")

# Goal / time parameters
GOAL_DIS = 0.2
STOP_SPEED = 0.0   # m/s
MAX_TIME = 500.0

# Vehicle constraints
TARGET_SPEED = 5.0 / 3.6
MAX_ITER = 3
DU_TH = 0.1

# Vehicle geometry
LENGTH = 1
WIDTH = 0.5
WB = 0.9  # Wheelbase
MAX_STEER = np.deg2rad(45.0)   # Maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # Maximum steering speed [rad/s]
MAX_SPEED = 10 / 3.6
MIN_SPEED = -20.0 / 3.6
MAX_ACCEL = 3.0  # m/s^2

# Robot bounding circle
ROBOT_RADIUS = math.hypot(LENGTH / 2, WIDTH / 2)

# Additional cost weighting for distance
DISTANCE_WEIGHT = 100


#If show_animation = True, the the plot is updated in real-time for each MPC iteration.
#If set to False, the code runs silently without showing intermediate plots.
show_animation = True

###############################################################################
# 2) HELPER CLASSES & FUNCTIONS
###############################################################################

class State:
    """Vehicle state class."""
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

def angle_mod(x, zero_2_2pi=False, degree=False):
    """Utility for normalizing angles."""
    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)
    if zero_2_2pi:
        mod_angle = x % (2 * math.pi)
    else:
        mod_angle = (x + math.pi) % (2 * math.pi) - math.pi
    if degree:
        mod_angle = np.rad2deg(mod_angle)
    return mod_angle

def plot_truck(ax, state):
    """
    Plot a rectangle of size LENGTH x WIDTH at the state's position and yaw.
    Also, plot the robot's bounding circle as a dashed line.
    """
    cx, cy, yaw = state.x, state.y, state.yaw

    # corners in local coordinates
    corners_local = np.array([
        [ +LENGTH/2, +WIDTH/2 ],
        [ +LENGTH/2, -WIDTH/2 ],
        [ -LENGTH/2, -WIDTH/2 ],
        [ -LENGTH/2, +WIDTH/2 ],
    ])
    # rotation
    R_2d = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw),  math.cos(yaw)]
    ])
    corners_world = (R_2d @ corners_local.T).T + np.array([cx, cy])
    corners_world = np.vstack([corners_world, corners_world[0,:]])  # close the rectangle

    # Plot the vehicle rectangle
    ax.plot(corners_world[:,0], corners_world[:,1], 'k-', linewidth=2)

    # # Plot the robot's bounding circle
    # circle = plt.Circle((cx, cy), ROBOT_RADIUS, color='gray', linestyle='--', fill=False)
    # ax.add_artist(circle)

def plot_time_series(time_log, traj_x, traj_y, traj_yaw, traj_v, mpc_accel_applied, mpc_steer_applied):
    """
    Plots:
    1) x and y vs time
    2) yaw and velocity vs time
    3) acceleration and steering vs time
    in separate figures or subplots.
    """

    # 1) x and y vs. time
    plt.figure()
    plt.plot(time_log, traj_x, label="x(t)")
    plt.plot(time_log, traj_y, label="y(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("X and Y vs. Time")
    plt.legend()
    plt.grid(True)

    # 2) yaw and velocity vs. time
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time_log, traj_yaw, 'b', label="Yaw")
    plt.xlabel("Time [s]")
    plt.ylabel("Yaw [rad]")
    plt.title("Yaw vs. Time")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(time_log, traj_v, 'r', label="Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("v [m/s]")
    plt.title("Velocity vs. Time")
    plt.grid(True)

    plt.tight_layout()

    # 3) acceleration and steering vs. time
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time_log, mpc_accel_applied, 'g', label="Accel a(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Accel [m/s^2]")
    plt.title("Acceleration vs. Time")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(time_log, mpc_steer_applied, 'm', label="Steer phi(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Steer [rad]")
    plt.title("Steering vs. Time")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


###############################################################################
# 3) SPLINE FUNCTIONS
###############################################################################

def calc_spline_course(x, y, ds=0.1):
    """
    Calculate a cubic spline through points (x, y) with sampling ds.
    """
    class CubicSpline2D:
        def __init__(self, x, y):
            dx_ = np.diff(x)
            dy_ = np.diff(y)
            self.s = np.hstack(([0], np.cumsum(np.hypot(dx_, dy_))))
            self.sx = CubicSpline1D(self.s, x)
            self.sy = CubicSpline1D(self.s, y)

        def calc_position(self, s):
            return self.sx.calc_position(s), self.sy.calc_position(s)

        def calc_curvature(self, s):
            dx_ = self.sx.calc_first_derivative(s)
            dy_ = self.sy.calc_first_derivative(s)
            ddx_ = self.sx.calc_second_derivative(s)
            ddy_ = self.sy.calc_second_derivative(s)
            return (ddy_ * dx_ - ddx_ * dy_) / ((dx_**2 + dy_**2)**1.5)

        def calc_yaw(self, s):
            dx_ = self.sx.calc_first_derivative(s)
            dy_ = self.sy.calc_first_derivative(s)
            return math.atan2(dy_, dx_)

    class CubicSpline1D:
        def __init__(self, x, y):
            h = np.diff(x)
            self.a = y
            self.c = np.zeros_like(y)
            A = np.zeros((len(x), len(x)))
            B = np.zeros(len(x))

            for i in range(1, len(x) - 1):
                A[i, i - 1] = h[i - 1]
                A[i, i]     = 2 * (h[i - 1] + h[i])
                A[i, i + 1] = h[i]
                B[i] = 3 * ((y[i + 1] - y[i]) / h[i]
                          - (y[i] - y[i - 1]) / h[i - 1])
            A[0, 0] = 1
            A[-1, -1] = 1
            self.c = np.linalg.solve(A, B)
            self.b = (np.diff(self.a) / h
                      - h * (2*self.c[:-1] + self.c[1:]) / 3)
            self.d = np.diff(self.c) / (3 * h)
            self.x = x

        def calc_position(self, xx):
            i = np.searchsorted(self.x, xx) - 1
            i = max(0, min(i, len(self.a)-1))
            dx_ = xx - self.x[i]
            return self.a[i] + self.b[i]*dx_ + self.c[i]*dx_**2 + self.d[i]*dx_**3

        def calc_first_derivative(self, xx):
            i = np.searchsorted(self.x, xx) - 1
            i = max(0, min(i, len(self.a)-1))
            dx_ = xx - self.x[i]
            return self.b[i] + 2*self.c[i]*dx_ + 3*self.d[i]*dx_**2

        def calc_second_derivative(self, xx):
            i = np.searchsorted(self.x, xx) - 1
            i = max(0, min(i, len(self.a)-1))
            dx_ = xx - self.x[i]
            return 2*self.c[i] + 6*self.d[i]*dx_

    sp = CubicSpline2D(x, y)
    s_array = np.arange(0, sp.s[-1], ds)
    rx, ry, ryaw, rk = [], [], [], []
    for s_ in s_array:
        ix, iy = sp.calc_position(s_)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(s_))
        rk.append(sp.calc_curvature(s_))

    return rx, ry, ryaw, rk, s_array

def remove_duplicate_points(xs, ys, eps=1e-6):
    """
    Remove consecutive duplicate or almost-duplicate points from (xs, ys).
    This prevents zero-length segments that cause a singular spline matrix.
    eps is the distance threshold under which points are considered 'the same'.
    """   
    if len(xs) < 2:
        return xs, ys
    new_x = [xs[0]]
    new_y = [ys[0]]
    for i in range(1, len(xs)):
        dx_ = xs[i] - new_x[-1]
        dy_ = ys[i] - new_y[-1]
        dist_ = math.hypot(dx_, dy_)
        if dist_ > eps:
            new_x.append(xs[i])
            new_y.append(ys[i])
    return np.array(new_x), np.array(new_y)

def get_switch_back_course(dl):
    """
    Creates a smooth spline from the 'best_path.csv' file, removing duplicates
    if necessary, and sampling at distance dl.
    """
    ax = best_path_array[:,0]
    ay = best_path_array[:,1]

    ax, ay = remove_duplicate_points(ax, ay)
    if len(ax) < 3:
        print("Warning: Not enough distinct points. Adding small offsets.")
        while len(ax) < 3:
            ax = np.append(ax, ax[-1]+0.0001)
            ay = np.append(ay, ay[-1]+0.0001)

    cx, cy, cyaw, ck, s_ = calc_spline_course(ax, ay, ds=dl)
    return cx, cy, cyaw, ck

###############################################################################
# 4) MPC OBSTACLE HANDLING
###############################################################################

def get_linear_model_matrix(v, theta, phi):
    """
    Build linearized system matrices A, B, and offset C around (v, theta, phi).
    """
    A = np.zeros((NX, NX))
    A[0,0] = 1.0
    A[1,1] = 1.0
    A[2,2] = 1.0
    A[3,3] = 1.0

    A[0,2] = DT * math.cos(theta)
    A[0,3] = -DT * v * math.sin(theta)
    A[1,2] = DT * math.sin(theta)
    A[1,3] = DT * v * math.cos(theta)
    A[3,2] = DT * math.tan(phi) / WB

    B = np.zeros((NX, NU))
    B[2,0] = DT
    B[3,1] = DT * v / (WB * (math.cos(phi)**2))

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(theta) * theta
    C[1] = -DT * v * math.cos(theta) * theta
    C[3] = v * phi / (WB * (math.cos(phi)**2))

    return A, B, C

def build_obstacle_linear_terms(xbar, all_obstacles):
    """
    For each time step k in {0,...,T-1}, build an array of shape [num_obstacles, 3]
    describing the half-plane constraint:

        n_k^T [x_k, y_k]^T <= b_k

    Where:
      - n_k = ( (x_i - x_t), (y_i - y_t) ) / norm(...)
      - r_total = obs_radius + ROBOT_RADIUS
      - q_i = (x_i, y_i) - r_total * n_k
      - b_k = n_k^T q_i

    xbar shape is (NX, T+1); 
      xbar[0,k] is the predicted x of the robot at time k
      xbar[1,k] is the predicted y of the robot at time k
      ...
    """
    obstacle_linear_terms = []

    for k in range(T):
        # The predicted robot position at time step k
        robot_x = xbar[0, k]
        robot_y = xbar[1, k]

        terms_k = []
        for (obs_x, obs_y, obs_r) in all_obstacles:
            # 1) Combined obstacle radius, the sum of obstacle radius & vehicle radius
            r_total = obs_r + ROBOT_RADIUS

            # 2) Construct the vector from the robot to the obstacle center
            nx_ = obs_x - robot_x
            ny_ = obs_y - robot_y
            dist = math.hypot(nx_, ny_)

            # If the distance is extremely small, we fallback to (1,0) to avoid dividing by zero
            if dist < 1e-6:
                nx_, ny_ = 1.0, 0.0
                dist = 1.0

            # Normalize n_k to have length 1
            nx_ /= dist
            ny_ /= dist

            # 3) The boundary point q_i is the obstacle center minus r_total along n_k
            qx = obs_x - r_total * nx_
            qy = obs_y - r_total * ny_

            # 4) b_k = n_k^T q_i  =>  This sets up n_k^T p(t) <= b_k
            b_ = nx_ * qx + ny_ * qy

            terms_k.append([nx_, ny_, b_])

        # After building all obstacles' constraints at time k, store them
        obstacle_linear_terms.append(np.array(terms_k))

    return obstacle_linear_terms


def linear_mpc_control_with_obstacles(xref, xbar, x0, dref, obstacle_linear_terms):
    """
    Solve the MPC problem with full obstacle constraints for all obstacles.
    """
    x = cvxpy.Variable((NX, T+1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        # 1) Cost on control input
        cost += cvxpy.quad_form(u[:, t], R)

        # 2) State tracking cost
        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        # 3) System dynamics constraint: x_{t+1} = A*x_t + B*u_t + C
        A, B, C = get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t+1] == A @ x[:, t] + B @ u[:, t] + C]

        # 4) Additional cost on input rate and distance traveled
        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t+1] - u[:, t], Rd)
            constraints += [
                cvxpy.abs(u[1, t+1] - u[1, t]) <= MAX_DSTEER * DT
            ]
            # distance term
            cost += DISTANCE_WEIGHT * (
                (x[0, t+1] - x[0, t])**2 +
                (x[1, t+1] - x[1, t])**2
            )

        # 5) Full obstacle constraints
        obs_terms = obstacle_linear_terms[t]
        for i_obs in range(obs_terms.shape[0]):
            nx_ = obs_terms[i_obs, 0]
            ny_ = obs_terms[i_obs, 1]
            b_  = obs_terms[i_obs, 2]
            constraints += [
                nx_ * x[0, t] + ny_ * x[1, t] <= b_
            ]

    # 6) Final cost on final state
    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    # 7) Additional constraints
    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    # 8) Solve
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(
        solver=cvxpy.OSQP,
        verbose=True,
        max_iter=20000,
        eps_abs=1e-3,
        eps_rel=1e-3
    )

    if prob.status in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
        ox = np.array(x.value[0, :]).flatten()
        oy = np.array(x.value[1, :]).flatten()
        ov = np.array(x.value[2, :]).flatten()
        oyaw = np.array(x.value[3, :]).flatten()
        oa = np.array(u.value[0, :]).flatten()
        ophi = np.array(u.value[1, :]).flatten()
    else:
        print("MPC could not solve with full obstacles.")
        ox, oy, ov, oyaw, oa, ophi = None, None, None, None, None, None

    return oa, ophi, ox, oy, oyaw, ov

def update_state(state, a, phi):
    """
    Kinematic bicycle update.
    """
    if phi >= MAX_STEER:
        phi = MAX_STEER
    elif phi <= -MAX_STEER:
        phi = -MAX_STEER

    state.x += state.v * math.cos(state.yaw) * DT
    state.y += state.v * math.sin(state.yaw) * DT
    state.yaw += state.v / WB * math.tan(phi) * DT
    state.v += a * DT

    # saturations
    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state

def check_goal(state, goal):
    """
    Check if close enough to goal in XY and near zero speed, if needed.
    """
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)
    return d <= GOAL_DIS

def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl):
    """
    Build a reference trajectory (xref) for T steps from the global path.
    """
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    # find the closest index
    dx_ = [state.x - icx for icx in cx]
    dy_ = [state.y - icy for icy in cy]
    d_ = np.hypot(dx_, dy_)
    ind = np.argmin(d_)

    travel = 0.0
    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[-1]
            xref[1, i] = cy[-1]
            xref[2, i] = sp[-1]
            xref[3, i] = cyaw[-1]
            dref[0, i] = 0.0

    return xref, ind, dref

def shift_mpc_solution(ox, oy, ov, oyaw, oa, ophi):
    """
    Enable the receiding horizon by shifting the MPC solution 
    forward by 1 step for the next iteration.
    """
    oxn = np.concatenate([ox[1:], [ox[-1]]], axis=0)
    oyn = np.concatenate([oy[1:], [oy[-1]]], axis=0)
    ovn = np.concatenate([ov[1:], [ov[-1]]], axis=0)
    oyawn = np.concatenate([oyaw[1:], [oyaw[-1]]], axis=0)

    oan = np.concatenate([oa[1:], [oa[-1]]], axis=0)
    ophin = np.concatenate([ophi[1:], [ophi[-1]]], axis=0)

    # Build the new initial guess for the next iteration
    xbar = np.vstack([oxn, oyn, ovn, oyawn])
    return xbar, oan, ophin

def do_simulation_with_obstacles(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    Perform the MPC simulation with all obstacles in constraints.
    At the end, compute & print:
      - total Euclidean distance traveled
      - sum of absolute accelerations
      - sum of absolute steering inputs
      - save the states to CSV

    Returns:
      (traj_x, traj_y, traj_yaw, traj_v, time_log, mpc_accel_extended, mpc_steer_extended)
    so that shapes match time_log exactly.
    """
    goal = [cx[-1], cy[-1]]
    state = initial_state

    # time
    time_ = 0.0
    time_log = [time_]  # for storing time at each iteration

    # for logging states
    traj_x = [state.x]
    traj_y = [state.y]
    traj_yaw = [state.yaw]
    traj_v = [state.v]

    # record controls
    mpc_accel_applied = []
    mpc_steer_applied = []

    # xbar guess
    xbar = np.tile(np.array([state.x, state.y, state.v, state.yaw]).reshape(-1, 1), (1, T + 1))

    while time_ <= MAX_TIME:
        xref, ind, dref = calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl)
        x0 = np.array([state.x, state.y, state.v, state.yaw])

        # Build obstacle constraints for all obstacles
        obstacle_terms = build_obstacle_linear_terms(xbar, obstacles_all)

        oa, ophi, ox, oy, oyaw_, ov_ = linear_mpc_control_with_obstacles(
            xref, xbar, x0, dref, obstacle_terms
        )
        if oa is None:
            print("No feasible solution at time=", time_)
            break

        # apply first control
        a_cmd = oa[0]
        phi_cmd = ophi[0]

        mpc_accel_applied.append(a_cmd)
        mpc_steer_applied.append(phi_cmd)

        state = update_state(state, a_cmd, phi_cmd)

        xbar_new = np.vstack([ox, oy, ov_, oyaw_])
        xbar_shifted, oa_shifted, ophi_shifted = shift_mpc_solution(
            ox, oy, ov_, oyaw_, oa, ophi
        )
        xbar = xbar_shifted

        time_ += DT
        time_log.append(time_)

        traj_x.append(state.x)
        traj_y.append(state.y)
        traj_yaw.append(state.yaw)
        traj_v.append(state.v)

        if check_goal(state, goal):
            print(f"Goal Reached at time = {time_:.2f} s")
            break

        if show_animation:
            plt.cla()
            plt.plot(cx, cy, '-r', label='Course')
            # obstacles
            for obs in obstacles_for_display:
                circle = plt.Circle((obs[0], obs[1]), obs[2], color='b', alpha=0.3)
                plt.gca().add_artist(circle)

                r_total = obs[2] + ROBOT_RADIUS  
                safety_circle = plt.Circle(
                    (obs[0], obs[1]),
                    r_total,
                    color='gray',
                    linestyle='--',
                    fill=False,
                    alpha=0.7
                )
                plt.gca().add_artist(safety_circle)

            plt.plot(traj_x, traj_y, '-g', label='MPC Path')
            plt.scatter([state.x], [state.y], color='green', s=50, marker='o')
            plot_truck(plt.gca(), state)
            plt.axis('equal')
            plt.pause(0.001)

    # compute final stats
    total_distance = 0.0
    for i in range(1, len(traj_x)):
        dx_ = traj_x[i] - traj_x[i - 1]
        dy_ = traj_y[i] - traj_y[i - 1]
        total_distance += math.hypot(dx_, dy_)

    total_abs_accel = sum(abs(a) for a in mpc_accel_applied)
    total_abs_steer = sum(abs(s) for s in mpc_steer_applied)

    print(f"Final path distance traveled: {total_distance:.3f} m")
    print(f"Total absolute acceleration used: {total_abs_accel:.3f}")
    print(f"Total absolute steering used: {total_abs_steer:.3f}")

    # Now we fix the length mismatch by extending the control arrays to match time_log
    if len(mpc_accel_applied) < len(time_log):
        mpc_accel_applied.append(0.0)
    if len(mpc_steer_applied) < len(time_log):
        mpc_steer_applied.append(0.0)

    # Save to CSV
    data_array = np.column_stack([
        time_log,
        traj_x,
        traj_y,
        traj_yaw,
        traj_v,
        mpc_accel_applied,
        mpc_steer_applied
    ])

    np.savetxt(
        "MPC_path.csv",
        data_array,
        delimiter=",",
        header="time,x,y,yaw,v,a,phi",
        comments=""
    )
    print("Saved MPC path data to 'MPC_path.csv'")

    # Return final states and logs
    return traj_x, traj_y, traj_yaw, traj_v, time_log, mpc_accel_applied, mpc_steer_applied

def main():
    dl = 0.1
    cx, cy, cyaw, ck = get_switch_back_course(dl)
    print(f"Course generated: {len(cx)} points")

    sp = [TARGET_SPEED] * len(cx)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[1], v=0.0)

    # Convert obstacles_for_display to a list of tuples for constraints
    global obstacles_all
    obstacles_all = [(o[0], o[1], o[2]) for o in obstacles_for_display]

    x, y, yaw_, v_, time_log, accel_log, steer_log = do_simulation_with_obstacles(
        cx, cy, cyaw, ck, sp, dl, initial_state
    )

    # Plot the time series data
    plot_time_series(time_log, x, y, yaw_, v_, accel_log, steer_log)

    if show_animation:
        plt.figure()
        plt.plot(cx, cy, '-r', label='Course')
        for obs in obstacles_for_display:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='b', alpha=0.3)
            plt.gca().add_artist(circle)
        plt.plot(x, y, '-g', label='MPC Path')
        plt.axis('equal')
        plt.legend()
        plt.title("MPC with Full Obstacle Avoidance")
        plt.show()

if __name__ == "__main__":
    main()