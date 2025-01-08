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
# path_path = os.path.join(script_dir, "..", "Gijs_code", "Random.csv")
path_path = os.path.join(script_dir, "..", "Gijs_code", "best_path.csv")

best_path_array = pd.read_csv(path_path)
best_path_array = np.array(best_path_array)

# Obstacles from CSV (for visualization ONLY, not used in constraints)
obstacles_path = os.path.join(script_dir, "..", "Gijs_code", "obstacles.csv")
obstacles_for_display = np.array(pd.read_csv(obstacles_path))

# PROBLEM SETUP
NX = 4  
NU = 2  
T = 100 
DT = 0.05


R = np.diag([0.01, 0.01])
Rd = np.diag([0.01, 1.0])
Q = np.diag([0.001, 0.001, 0.1, 0.1])
Qf = 10*Q

print(f"Using T={T}, DT={DT}, total horizon = {T*0.03}s")

GOAL_DIS = 0.2
STOP_SPEED = 0/3.6
MAX_TIME = 500.0
TARGET_SPEED =50.0/3.6
MAX_ITER = 3
DU_TH = 0.1
LENGTH = 0.5
WIDTH = 0.25
WB = 0.5  # Wheelbase 
MAX_STEER = np.deg2rad(45.0)  # Maximum steering angle in radians
MAX_DSTEER = np.deg2rad(30.0)  # Maximum steering speed in radians per second
MAX_SPEED = 50/3.6
MIN_SPEED = -20.0/3.6
MAX_ACCEL = 3.0  # Maximum acceleration/deceleration in m/s^2

ROBOT_RADIUS = math.hypot(LENGTH / 2, WIDTH / 2)
DISTANCE_WEIGHT = 100 # Weight for distance optimization

show_animation = True

# Define the single obstacle
obstacles = [(2, 0, 0.5)]  # Single red obstacle at (2,0) with radius 0.5

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

def angle_mod(x, zero_2_2pi=False, degree=False):
    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)
    if zero_2_2pi:
        mod_angle = x % (2*math.pi)
    else:
        mod_angle = (x+math.pi) % (2*math.pi) - math.pi
    if degree:
        mod_angle = np.rad2deg(mod_angle)
    return mod_angle    


def plot_truck(ax, state):
    """Plot a rectangle of size LENGTH x WIDTH at (state.x, state.y, yaw)."""
    cx, cy, yaw = state.x, state.y, state.yaw

    # corners in local coordinates:  +x forward, +y left (for example)
    corners_local = np.array([
        [+LENGTH/2, +WIDTH/2],
        [+LENGTH/2, -WIDTH/2],
        [-LENGTH/2, -WIDTH/2],
        [-LENGTH/2, +WIDTH/2],
    ])
    # rotation
    R = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw),  math.cos(yaw)]
    ])
    corners_world = (R @ corners_local.T).T + np.array([cx, cy])

    # close the rectangle
    corners_world = np.vstack([corners_world, corners_world[0,:]])

    ax.plot(corners_world[:,0], corners_world[:,1], 'k-', linewidth=2)


###############################################################################
# 3) SPLINE FUNCTIONS
###############################################################################

def calc_spline_course(x, y, ds=0.1):
    class CubicSpline2D:
        def __init__(self, x, y):
            dx = np.diff(x)
            dy = np.diff(y)
            self.s = np.hstack(([0], np.cumsum(np.hypot(dx, dy))))
            self.sx = CubicSpline1D(self.s, x)
            self.sy = CubicSpline1D(self.s, y)

        def calc_position(self, s):
            x = self.sx.calc_position(s)
            y = self.sy.calc_position(s)
            return x, y

        def calc_curvature(self, s):
            dx = self.sx.calc_first_derivative(s)
            dy = self.sy.calc_first_derivative(s)
            ddx = self.sx.calc_second_derivative(s)
            ddy = self.sy.calc_second_derivative(s)
            return (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))

        def calc_yaw(self, s):
            dx = self.sx.calc_first_derivative(s)
            dy = self.sy.calc_first_derivative(s)
            return math.atan2(dy, dx)

    class CubicSpline1D:
        def __init__(self, x, y):
            h = np.diff(x)
            self.a = y
            self.c = np.zeros_like(y)
            A = np.zeros((len(x), len(x)))
            B = np.zeros(len(x))

            for i in range(1, len(x) - 1):
                A[i, i - 1] = h[i - 1]
                A[i, i] = 2 * (h[i - 1] + h[i])
                A[i, i + 1] = h[i]
                B[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

            A[0, 0] = 1
            A[-1, -1] = 1
            self.c = np.linalg.solve(A, B)
            self.b = np.diff(self.a) / h - h * (2 * self.c[:-1] + self.c[1:]) / 3
            self.d = np.diff(self.c) / (3 * h)
            self.x = x

        def calc_position(self, x):
            i = np.searchsorted(self.x, x) - 1
            i = max(0, min(i, len(self.a) - 1))
            dx = x - self.x[i]
            return self.a[i] + self.b[i] * dx + self.c[i] * dx**2 + self.d[i] * dx**3

        def calc_first_derivative(self, x):
            i = np.searchsorted(self.x, x) - 1
            dx = x - self.x[i]
            return self.b[i] + 2 * self.c[i] * dx + 3 * self.d[i] * dx**2

        def calc_second_derivative(self, x):
            i = np.searchsorted(self.x, x) - 1
            dx = x - self.x[i]
            return 2 * self.c[i] + 6 * self.d[i] * dx

    sp = CubicSpline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


def remove_duplicate_points(xs, ys, eps=1e-6):
    """
    Remove consecutive duplicate or almost-duplicate points from (xs, ys).
    This prevents zero-length segments that cause a singular spline matrix.
    eps is the distance threshold under which points are considered 'the same'.
    """
    if len(xs) < 2:
        return xs, ys  # nothing to remove

    new_x = [xs[0]]
    new_y = [ys[0]]
    for i in range(1, len(xs)):
        dx = xs[i] - new_x[-1]
        dy = ys[i] - new_y[-1]
        dist = math.hypot(dx, dy)
        if dist > eps:
            new_x.append(xs[i])
            new_y.append(ys[i])
    return np.array(new_x), np.array(new_y)


def get_switch_back_course(dl):
    # Original RRT* path from CSV
    ax = best_path_array[:, 0]
    ay = best_path_array[:, 1]

    # 1) Remove duplicates
    ax, ay = remove_duplicate_points(ax, ay)

    # 2) If fewer than 3 points remain, add small offsets
    #    to avoid degeneracy in the spline
    if len(ax) < 3:
        print("Warning: Not enough distinct points. Adding small offsets.")
        while len(ax) < 3:
            ax = np.append(ax, ax[-1] + 0.0001)
            ay = np.append(ay, ay[-1] + 0.0001)

    # 3) Now safely call the spline
    cx, cy, cyaw, ck, s = calc_spline_course(ax, ay, ds=dl)
    return cx, cy, cyaw, ck

###############################################################################
# MODEL, OBSTACLE ETC.
###############################################################################

def get_linear_model_matrix(v, phi, delta):
    """
    Build linearized system matrices A, B, and offset C around
    (v, phi, delta).
    """
    A = np.matrix(np.zeros((NX, NX)))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0

    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = -DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.matrix(np.zeros((NX, NU)))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = -DT * v * math.cos(phi) * phi
    C[3] = v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C

def build_obstacle_linear_terms(xbar, obstacles):
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
        for (obs_x, obs_y, obs_r) in obstacles:
            # 1) Combined obstacle radius
            r_total = obs_r + ROBOT_RADIUS

            # 2) Normal vector from (robot_x, robot_y) -> (obs_x, obs_y)
            nx_ = obs_x - robot_x
            ny_ = obs_y - robot_y
            dist = math.hypot(nx_, ny_)

            # If dist ~ 0, fallback normal to x direction
            if dist < 1e-6:
                nx_, ny_ = 1.0, 0.0
                dist = 1.0
            
            # Normalize
            nx_ /= dist
            ny_ /= dist

            # 3) The boundary point q_i
            qx = obs_x - r_total * nx_
            qy = obs_y - r_total * ny_

            # 4) b_k = n_k^T q_i
            b_ = nx_ * qx + ny_ * qy

            terms_k.append([nx_, ny_, b_])

        obstacle_linear_terms.append(np.array(terms_k))

    return obstacle_linear_terms

def linear_mpc_control_with_obstacles(xref, xbar, x0, dref, obstacle_linear_terms):
    x = cvxpy.Variable((NX, T+1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        # Cost function control effort 
        cost += cvxpy.quad_form(u[:, t], R)
        # Reference tracking, deviation from states
        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q) 

        # Dynamic constraints
        A, B, C = get_linear_model_matrix(xbar[2,t], xbar[3,t], dref[0,t])
        constraints += [x[:, t+1] == A@x[:, t] + B@u[:, t] + C]

        if t < T-1:
            cost += cvxpy.quad_form(u[:, t+1]-u[:, t], Rd)  # Penalizing deviation control input
            constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= MAX_DSTEER*DT]   # Constrain
            # Distance optimization cost
            cost += DISTANCE_WEIGHT * (
                (x[0, t+1] - x[0, t])**2 +
                (x[1, t+1] - x[1, t])**2
            )

        # Obstacle avoidance constraints    
        nx_ = obstacle_linear_terms[t][0,0]
        ny_ = obstacle_linear_terms[t][0,1]
        b_  = obstacle_linear_terms[t][0,2] 
        constraints += [nx_*x[0,t] + ny_*x[1,t] <= b_] # Constraint for only one obstacle

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)
    constraints += [x[:,0] == x0]
    constraints += [x[2,:] <= MAX_SPEED]
    constraints += [x[2,:] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0,:]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1,:]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    # prob.solve(solver=cvxpy.OSQP, verbose=True)
    prob.solve(
    solver=cvxpy.OSQP,
    verbose=True,
    max_iter=20000,
    eps_abs=1e-3,
    eps_rel=1e-3
)

    if prob.status in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
        ox = np.array(x.value[0,:]).flatten()
        oy = np.array(x.value[1,:]).flatten()
        ov = np.array(x.value[2,:]).flatten()
        oyaw = np.array(x.value[3,:]).flatten()
        oa = np.array(u.value[0,:]).flatten()
        odelta = np.array(u.value[1,:]).flatten()
    else:
        print("MPC could not solve with single obstacle.")
        ox, oy, ov, oyaw, oa, odelta = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov

def update_state(state, a, delta):
    """Kinematic bicycle update."""
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x += state.v * math.cos(state.yaw) * DT
    state.y += state.v * math.sin(state.yaw) * DT
    state.yaw += state.v / WB * math.tan(delta) * DT
    state.v += a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state


def check_goal(state, goal):
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)
    return d <= GOAL_DIS

def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl):
    """
    Builds a reference trajectory xref for T steps 
    but does NOT do naive shifting around obstacles.
    """
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    # Find closest index
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
            # if beyond end
            xref[0, i] = cx[-1]
            xref[1, i] = cy[-1]
            xref[2, i] = sp[-1]
            xref[3, i] = cyaw[-1]
            dref[0, i] = 0.0

    return xref, ind, dref

def shift_mpc_solution(ox, oy, ov, oyaw, oa, odelta):
    """
    Shift the solution forward by 1 step (receding horizon).
    The new predicted trajectory xbar for the next iteration
    will be the old solution's 1->T steps + the last step repeated.
    """
    # shift states
    oxn = np.concatenate([ox[1:], [ox[-1]]], axis=0)
    oyn = np.concatenate([oy[1:], [oy[-1]]], axis=0)
    ovn = np.concatenate([ov[1:], [ov[-1]]], axis=0)
    oyawn = np.concatenate([oyaw[1:], [oyaw[-1]]], axis=0)

    # shift controls
    oan = np.concatenate([oa[1:], [oa[-1]]], axis=0)
    odeltan = np.concatenate([odelta[1:], [odelta[-1]]], axis=0)

    xbar = np.vstack([oxn, oyn, ovn, oyawn])
    return xbar, oan, odeltan

def do_simulation_with_obstacle(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    We ignore obstacles_for_display for constraints,
    but we still show them as "blue circles" on the plot for environment context.
    We DO impose constraints only for the single red obstacle at (2,0).
    """
    goal = [cx[-1], cy[-1]]
    state = initial_state

    time_ = 0.0
    traj_x, traj_y = [state.x], [state.y]
    traj_yaw, traj_v = [state.yaw], [state.v]

    # xbar guess
    xbar = np.tile(np.array([state.x, state.y, state.v, state.yaw]).reshape(-1,1), (1,T+1))
    # xbar = np.copy(xref)



    while time_ <= MAX_TIME:
        xref, ind, dref = calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl)
        x0 = np.array([state.x, state.y, state.v, state.yaw])

        # Build obstacle constraint
        single_obstacle_terms = build_obstacle_linear_terms(xbar, obstacles)

        oa, odelta, ox, oy, oyaw_, ov_ = linear_mpc_control_with_obstacles(
            xref, xbar, x0, dref, single_obstacle_terms
        )
        if oa is None:
            print("No feasible solution at time=", time_)
            break

        # Apply first control
        a_cmd = oa[0]
        delta_cmd = odelta[0]
        state = update_state(state, a_cmd, delta_cmd)

        # Shift horizon
        xbar_new = np.vstack([ox, oy, ov_, oyaw_])
        xbar_shifted, oa_shifted, odelta_shifted = shift_mpc_solution(
            ox, oy, ov_, oyaw_, oa, odelta
        )
        xbar = xbar_shifted

        time_ += DT
        traj_x.append(state.x)
        traj_y.append(state.y)
        traj_yaw.append(state.yaw)
        traj_v.append(state.v)
        if check_goal(state, goal):
            print("Goal Reached at time:", time_)
            break

        if show_animation:
            plt.cla()
            plt.plot(cx, cy, '-r', label='Course')

            # Plot the original environment obstacles in blue (no constraints)
            for obs in obstacles_for_display:
                circle = plt.Circle((obs[0], obs[1]), obs[2], color='b', alpha=0.3)
                plt.gca().add_artist(circle)

            # Dynamically plot the red obstacles from the `obstacles` list
            for obs in obstacles:
                circle_red = plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.8)
                plt.gca().add_artist(circle_red)

            plt.plot(traj_x, traj_y, '-g', label='MPC Path')
            plt.scatter([state.x], [state.y], color='green', s=50, marker='o')

            # Plot the rectangle truck
            plot_truck(plt.gca(), state)
            plt.axis('equal')
            plt.pause(0.001)
    return traj_x, traj_y, traj_yaw, traj_v


def main():
    dl = 0.1
    cx, cy, cyaw, ck = get_switch_back_course(dl)
    print(f"Course generated: {len(cx)} points")

    sp = [TARGET_SPEED]*len(cx)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[1], v=0.0)
    # obstacles is already defined as a global variable at the start of the code

    # Notice we do NOT pass obstacles_for_display to do_simulation
    # Instead, do_simulation will build a single obstacle constraint at (2,0).
    x, y, yaw_, v_ = do_simulation_with_obstacle(cx, cy, cyaw, ck, sp, dl, initial_state)

    if show_animation:
        plt.figure()
        plt.plot(cx, cy, '-r', label='Course')
        # show environment obstacles
        for obs in obstacles_for_display:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='b', alpha=0.3)
            plt.gca().add_artist(circle)
        # single red obstacle
        for obs in obstacles:
            circle_red = plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.8)
            plt.gca().add_artist(circle_red)

        plt.plot(x, y, '-g', label='MPC Path')
        plt.axis('equal')
        plt.legend()
        plt.title("MPC with Single Red Obstacle")
        plt.show()

if __name__ == "__main__":
    main()
