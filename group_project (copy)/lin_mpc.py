import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from model import export_kinematic_bicycle_model
from cubic_trajectory import generate_cubic_spline_trajectory

# Constants and parameters
NX = 4  # x = [x, y, v, yaw]
NU = 2  # u = [accel, steer]
T = 5  # horizon length

# MPC parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = 10 * np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
DT = 0.2  # time step [s]
WB = 2.5  # wheelbase [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum acceleration [m/s^2]

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, a, delta):
        delta = np.clip(delta, -MAX_STEER, MAX_STEER)
        self.x += self.v * np.cos(self.yaw) * DT
        self.y += self.v * np.sin(self.yaw) * DT
        self.yaw += self.v / WB * np.tan(delta) * DT
        self.v += a * DT
        self.v = np.clip(self.v, MIN_SPEED, MAX_SPEED)

def get_linear_model_matrix(v, phi, delta):
    A = np.eye(NX)
    A[0, 2] = DT * np.cos(phi)
    A[0, 3] = -DT * v * np.sin(phi)
    A[1, 2] = DT * np.sin(phi)
    A[1, 3] = DT * v * np.cos(phi)
    A[3, 2] = DT * np.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * np.cos(delta)**2)

    C = np.zeros(NX)
    return A, B, C

def linear_mpc_control(xref, xbar, x0, dref):
    import cvxpy

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

def calc_ref_trajectory(state, smooth_x, smooth_y, target_speed):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))

    n_course = len(smooth_x)
    dx = smooth_x - state.x
    dy = smooth_y - state.y
    dist = np.hypot(dx, dy)
    ind = np.argmin(dist)

    for i in range(T + 1):
        if (ind + i) < n_course:
            xref[0, i] = smooth_x[ind + i]
            xref[1, i] = smooth_y[ind + i]
            xref[2, i] = target_speed
            xref[3, i] = 0.0  # Assuming a flat trajectory for yaw
        else:
            xref[:, i] = xref[:, i - 1]

    return xref, ind, dref

def do_simulation(smooth_x, smooth_y, initial_state):
    state = initial_state
    target_speed = 10.0 / 3.6

    x, y, yaw, v = [state.x], [state.y], [state.yaw], [state.v]

    for _ in range(500):
        xref, _, dref = calc_ref_trajectory(state, smooth_x, smooth_y, target_speed)
        x0 = np.array([state.x, state.y, state.v, state.yaw])

        oa, odelta, _, _, _, _ = linear_mpc_control(xref, xref, x0, dref)

        if oa is not None and odelta is not None:
            state.update(oa[0], odelta[0])

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)

        if np.hypot(state.x - smooth_x[-1], state.y - smooth_y[-1]) <= 1.5:
            print("Goal Reached")
            break

    return x, y, yaw, v

def main():
    waypoints = [
        [-6, 2],
        [-3.5, 1.5],
        [-3.2, 1.4],
        [-1.8, 1.8],
        [0.2, 1.7],
        [2, -1],
        [3.5, -3],
        [6, -3.5]
    ]

    smooth_x, smooth_y, _ = generate_cubic_spline_trajectory(waypoints)

    initial_state = State(x=smooth_x[0], y=smooth_y[0], yaw=0.0, v=0.0)
    x, y, yaw, v = do_simulation(smooth_x, smooth_y, initial_state)

    plt.plot(smooth_x, smooth_y, "--r", label="Reference")
    plt.plot(x, y, "-g", label="Optimized Path")
    plt.scatter([p[0] for p in waypoints], [p[1] for p in waypoints], color="blue", label="Waypoints")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
