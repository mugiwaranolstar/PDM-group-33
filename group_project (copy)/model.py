import numpy as np
from casadi import SX, vertcat, cos, sin, tan
from acados_template import AcadosModel

# Vehicle parameters
L = 4.5  # Wheelbase length [m]

def export_kinematic_bicycle_model():
    """
    Export a kinematic bicycle model for ACADOS.

    State vector: [x, y, v, theta] -> where:
        x: X position,
        y: Y position,
        v: velocity,
        theta: heading angle.

    Control vector: [a, phi] -> where:
        a: acceleration,
        phi: steering angle.
    """

    # Define state variables
    x = SX.sym('x')        # X position
    y = SX.sym('y')        # Y position
    v = SX.sym('v')        # Velocity
    theta = SX.sym('theta')  # Heading angle

    # State vector
    states = vertcat(x, y, v, theta)

    # Define state derivatives
    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    v_dot = SX.sym('v_dot')
    theta_dot = SX.sym('theta_dot')

    states_dot = vertcat(x_dot, y_dot, v_dot, theta_dot)

    # Define control inputs
    a = SX.sym('a')        # Acceleration
    phi = SX.sym('phi')    # Steering angle

    # Control vector
    controls = vertcat(a, phi)

    # Nonlinear kinematic equations
    dx = v * cos(theta)
    dy = v * sin(theta)
    dv = a
    dtheta = v * tan(phi) / L

    # Explicit dynamics
    f_expl = vertcat(dx, dy, dv, dtheta)

    # Implicit dynamics
    f_impl = states_dot - f_expl

    # Create the ACADOS model
    model = AcadosModel()
    model.f_impl_expr = f_impl  # Implicit dynamics
    model.f_expl_expr = f_expl  # Explicit dynamics
    model.x = states            # States
    model.xdot = states_dot     # State derivatives
    model.u = controls          # Control inputs
    model.name = "kinematic_bicycle_model"

    return model
