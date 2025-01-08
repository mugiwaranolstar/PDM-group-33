import matplotlib.pyplot as plt
import time
import cvxpy
import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import os
import pandas as pd
import random


# Import path
script_dir = os.path.dirname(os.path.abspath(__file__))
path_path = os.path.join(script_dir, "..", "Gijs_code", "best_path.csv")
best_path_array = pd.read_csv(path_path)
best_path_array = np.array(best_path_array)
#Import obstacles
obstacles_path = os.path.join(script_dir, "..", "Gijs_code", "obstacles.csv")
obstacles_array = pd.read_csv(obstacles_path)
# obstacles_array = np.array(obstacles_array)

# Constants and parameters
NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 15  # horizon length

# MPC parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 0.2  # goal distance
STOP_SPEED = 0 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# Iterative parameter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.03  # [s] time tick

# Vehicle parameters
LENGTH = 1  # [m]
WIDTH = 0.5  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 0.78  # [m]

MAX_STEER = np.deg2rad(60.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(180.0)  # maximum steering speed [rad/s]
MAX_SPEED = 5.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -5.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

show_animation = True

# State class
class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


def angle_mod(x, zero_2_2pi=False, degree=False):
    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)
    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi
    if degree:
        mod_angle = np.rad2deg(mod_angle)
    return mod_angle


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


def get_switch_back_course(dl):
    # ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ax = best_path_array[:, 0]
    # ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    ay = best_path_array[:, 1]
    cx, cy, cyaw, ck, s = calc_spline_course(ax, ay, ds=dl)
    return cx, cy, cyaw, ck


def get_linear_model_matrix(v, phi, delta):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = -DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = -DT * v * math.cos(phi) * phi
    C[3] = -DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


def linear_mpc_control(xref, xbar, x0, dref):
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)
        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)
    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.OSQP, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = np.array(x.value[0, :]).flatten()
        oy = np.array(x.value[1, :]).flatten()
        ov = np.array(x.value[2, :]).flatten()
        oyaw = np.array(x.value[3, :]).flatten()
        oa = np.array(u.value[0, :]).flatten()
        odelta = np.array(u.value[1, :]).flatten()
    else:
        print("MPC: Cannot solve problem")
        ox, oy, ov, oyaw, oa, odelta = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov


def update_state(state, a, delta):
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
    return d <= GOAL_DIS #and abs(state.v) <= STOP_SPEED


def calc_ref_trajectory_with_obstacle(state, cx, cy, cyaw, ck, sp, dl, obstacles):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]
    d = np.hypot(dx, dy)
    ind = np.argmin(d)

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            x = cx[ind + dind]
            y = cy[ind + dind]

            # Controleer of het punt binnen een obstakel valt
            for obs in obstacles:
                obs_x, obs_y, obs_radius = obs
                if math.hypot(x - obs_x, y - obs_y) <= obs_radius:
                    x += obs_radius * 2  # Verplaats het punt simpelweg naar buiten
                    y += obs_radius * 2

            xref[0, i] = x
            xref[1, i] = y
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


def add_random_obstacles(cx, cy, num_obstacles=2):
    obstacles = []
    for _ in range(num_obstacles):

        index = random.randint(8, len(cx) - 8)
        x_pos = cx[index]
        y_pos = cy[index]
        
        obstacles.append((x_pos, y_pos, 0.5))  

    return obstacles




def do_simulation_with_obstacle(cx, cy, cyaw, ck, sp, dl, initial_state, obstacles):
    goal = [cx[-1], cy[-1]]
    state = initial_state

    time = 0.0
    x, y, yaw, v = [state.x], [state.y], [state.yaw], [state.v]

    while MAX_TIME >= time:
        xref, ind, dref = calc_ref_trajectory_with_obstacle(state, cx, cy, cyaw, ck, sp, dl, obstacles)
        x0 = [state.x, state.y, state.v, state.yaw]

        oa, odelta, _, _, _, _ = linear_mpc_control(xref, xref, x0, dref)

        if odelta is not None:
            di, ai = odelta[0], oa[0]
            state = update_state(state, ai, di)

        time += DT
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)

        if check_goal(state, goal):
            print("Goal Reached")
            state.v = STOP_SPEED
            break

        if show_animation:
            plt.cla()
            plt.plot(cx, cy, "-r", label="Course")
            for obs in obstacles:
                circle = plt.Circle((obs[0], obs[1]), obs[2], color='b', alpha=0.5)
                plt.gca().add_artist(circle)
            plt.plot(x, y, "-g", label="Path")
            plt.axis("equal")
            plt.pause(0.001)

    return x, y, yaw, v


def main():
    dl = 0.1
    cx, cy, cyaw, ck = get_switch_back_course(dl)
    print(f"Course generated: {len(cx)} points")

    sp = [TARGET_SPEED] * len(cx)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[1], v=0.0)

    # Obstakels toevoegen (x, y, radius)
    obstacles = np.array([
        [15.0, 10.0, 3.0],  # Obstacle at (15, 10) with radius 3
    ])

    # np.append(obstacles, obstacles_array, axis=0)
    # obstacles.append(obstacles_array)

    random_obstacles = add_random_obstacles(cx, cy)
    obstacles = np.concatenate([obstacles, obstacles_array, random_obstacles], axis=0)

    # np.append(obstacles, random_obstacles, axis=0)

    # obstacles.append(random_obstacles)

    x, y, yaw, v = do_simulation_with_obstacle(cx, cy, cyaw, ck, sp, dl, initial_state, obstacles)

    if show_animation:
        plt.figure()
        plt.plot(cx, cy, "-r", label="Course")
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='b', alpha=0.5)
            plt.gca().add_artist(circle)
        plt.plot(x, y, "-g", label="Path")
        plt.axis("equal")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
