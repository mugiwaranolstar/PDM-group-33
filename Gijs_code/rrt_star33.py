import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
import os


# Constants for the bicycle model
MAX_STEERING_ANGLE = np.pi / 4
MAX_VELOCITY = 1.0
DT = 0.1

# RRT Parameters
MAX_NODES = 600
GOAL_THRESHOLD = 0.5
MAX_CONNECTION_DISTANCE = 1.0
RADIUS = MAX_CONNECTION_DISTANCE * 2
NUM_INITIAL_BRANCHES = 6
GOAL_BIAS = 0.0
ENVIRONMENT_BOUNDS = 7

ROBOT_LENGTH = 1.0
ROBOT_WIDTH = 0.5

# --------------------
# Environment Creation
# --------------------
def create_environment2(goal):
    """Creates a PyBullet environment with larger obstacles, borders, and a goal marker."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    obstacles = [
        (2, 1, 0.5), (4, -2, 0.5), (-3, -3, 0.5),
        (1, -2, 0.5), (-4, 4, 0.5), (3, -4, 0.5),
        (5, 1, 0.5), (-5, -5, 0.5), (0, 5, 0.5),
        (0, -2, 0.5), (-4, 0, 0.5), (4, 4, 0.5),
        (-1, -1, 0.5), (0, 2, 0.5), (-1, -5, 0.5),
        (2, -2, 0.5), (0, 1.5, 0.5), (3, -5, 0.5),
        (2, -1.5, 0.5)
    ]
    for x, y, z in obstacles:
        p.loadURDF("cube.urdf", [x, y, z], globalScaling=1.0)

    for x in range(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS, 1):
        p.loadURDF("cube.urdf", [x, -ENVIRONMENT_BOUNDS, 0.5], globalScaling=1.0)
        p.loadURDF("cube.urdf", [x, ENVIRONMENT_BOUNDS, 0.5], globalScaling=1.0)

    for y in range(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS + 0, 1):
        p.loadURDF("cube.urdf", [-ENVIRONMENT_BOUNDS, y, 0.5], globalScaling=1.0)
        p.loadURDF("cube.urdf", [ENVIRONMENT_BOUNDS, y, 0.5], globalScaling=1.0)

    p.loadURDF("sphere_small.urdf", [goal[0], goal[1], GOAL_THRESHOLD], globalScaling=GOAL_THRESHOLD * 2)
    return obstacles


# -----------------
# Distance & Models
# -----------------
def compute_distance(node1, node2):
    """Euclidean distance in x-y space."""
    return np.linalg.norm(np.array(node1[:2]) - np.array(node2[:2]))


def bicycle_step(state, velocity, steering_angle, dt=DT):
    """Propagates the bicycle model one time step forward."""
    x, y, theta = state
    # Debugging statement:
    # print(f"[DEBUG] bicycle_step: Starting from (x={x:.2f}, y={y:.2f}, theta={theta:.2f}), "
    #       f"velocity={velocity:.2f}, steering_angle={steering_angle:.2f}")
    theta_new = theta + velocity * np.tan(steering_angle) * dt
    x_new = x + velocity * np.cos(theta_new) * dt
    y_new = y + velocity * np.sin(theta_new) * dt
    return x_new, y_new, theta_new


def get_robot_corners(state):
    """Returns the (x, y) positions of the four corners of the robot (in world coords)."""
    x, y, theta = state
    dx = ROBOT_LENGTH / 2
    dy = ROBOT_WIDTH / 2

    local_corners = np.array([
        [-dx, -dy],
        [-dx,  dy],
        [ dx,  dy],
        [ dx, -dy]
    ])

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    world_corners = np.dot(local_corners, rotation_matrix.T) + np.array([x, y])
    return world_corners


def is_collision_free(state, obstacles):
    """Checks if the robot (modeled as a rectangle) is collision-free at the given state."""
    robot_corners = get_robot_corners(state)
    robot_polygon = Polygon(robot_corners)
    for obs_x, obs_y, _ in obstacles:
        # Create a circular buffer region around each obstacle
        if robot_polygon.intersects(Point(obs_x, obs_y).buffer(0.5)):
            return False
    return True


def is_edge_collision_free(start, end, obstacles, steps=5):
    """Checks if the straight-line (in state-space) interpolation is collision-free."""
    for i in range(steps + 1):
        alpha = i / steps
        x = (1 - alpha) * start[0] + alpha * end[0]
        y = (1 - alpha) * start[1] + alpha * end[1]
        theta = (1 - alpha) * start[2] + alpha * end[2]
        intermediate_state = (x, y, theta)
        if not is_collision_free(intermediate_state, obstacles):
            return False
    return True


# -------------------
# KD-Tree Management
# -------------------
def add_node(new_state, nodes, kd_tree):
    """Adds a node to the node list and rebuilds the KD-tree."""
    nodes.append(new_state)
    kd_tree = cKDTree([node[:2] for node in nodes])
    return kd_tree


# ---------------------
# Path Utility Function
# ---------------------
def simplify_path(path, obstacles, steps=10):
    """
    Simplifies the path by skipping unnecessary waypoints, checking collision-free edges.
    """
    simplified_path = [path[0]]  # Start with the first node
    for i in range(len(path) - 1):
        for j in range(len(path) - 1, i, -1):
            if is_edge_collision_free(path[i], path[j], obstacles, steps):
                simplified_path.append(path[j])
                break
    return simplified_path


# ------------------------------------------------------------------
# New function to steer from nearest toward random state (bicycle)
# ------------------------------------------------------------------
def steer_toward(nearest, rand_state, obstacles, max_distance=MAX_CONNECTION_DISTANCE):
    """
    Tries to steer from 'nearest' toward 'rand_state' following bicycle kinematics.
    We choose a steering angle that points roughly toward rand_state, but clamp it by MAX_STEERING_ANGLE.
    Then we step forward in small increments until we reach or exceed 'max_distance', or encounter collision.
    """

    x_n, y_n, theta_n = nearest
    x_r, y_r, _ = rand_state
    direction_angle = np.arctan2((y_r - y_n), (x_r - x_n))  # desired heading
    angle_diff = direction_angle - theta_n

    # Normalize angle_diff into [-pi, pi]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    # Clamp steering angle
    steering_angle = np.clip(angle_diff, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

    # We'll move in small steps and check collision
    distance_covered = 0.0
    step_size = 0.2  # smaller step for better collision checking
    current_state = (x_n, y_n, theta_n)

    while distance_covered < max_distance:
        next_state = bicycle_step(current_state, MAX_VELOCITY, steering_angle, dt=DT)
        # Debugging statement:
        # print(f"[DEBUG] steer_toward: next_state={next_state}, distance_covered={distance_covered:.2f}")

        if not is_collision_free(next_state, obstacles):
            # Debugging statement:
            # print("[DEBUG] steer_toward: Collision detected, stopping extension.")
            return None  # collision occurred, return None

        current_state = next_state
        distance_covered += step_size

    return current_state


# ---------
# RRT* Core
# ---------
def rrt_star(start, goal, obstacles):
    # Data structures
    nodes = [start]
    parents = {tuple(start): None}
    costs = {tuple(start): 0}
    edge_ids = {}
    best_path = None

    kd_tree = cKDTree([start[:2]])  # KD-tree for quick nearest-neighbor search

    def precompute_distance(node1, node2):
        return compute_distance(node1, node2)

    # -----------------------------
    # Generate initial branches
    # -----------------------------
    for _ in range(NUM_INITIAL_BRANCHES):
        steering_angle = random.uniform(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
        branch_state = bicycle_step(start, MAX_VELOCITY, steering_angle)
        if is_collision_free(branch_state, obstacles):
            kd_tree = add_node(branch_state, nodes, kd_tree)
            parents[tuple(branch_state)] = start
            costs[tuple(branch_state)] = precompute_distance(branch_state, start)
            edge_ids[tuple(branch_state)] = p.addUserDebugLine(
                [start[0], start[1], 0.1],
                [branch_state[0], branch_state[1], 0.1],
                [1, 0, 0],
                lineWidth=1.0
            )

    # We store the goal in a tuple but do not add it as a regular node.
    goal_node = tuple(goal)

    # -----------------------------
    # Main RRT* Loop
    # -----------------------------
    for i in range(MAX_NODES):
        if i % 100 == 0:
            print(f"[DEBUG] Processing node {i} / {MAX_NODES}")

        # Sample a random state or bias toward the goal
        rand_state = goal if random.random() < GOAL_BIAS else (
            random.uniform(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS),
            random.uniform(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS),
            random.uniform(-np.pi, np.pi),
        )

        # Find the nearest node in the tree
        _, nearest_idx = kd_tree.query(rand_state[:2])
        nearest = nodes[nearest_idx]

        # Steer from 'nearest' toward 'rand_state' using bicycle constraints
        new_state = steer_toward(nearest, rand_state, obstacles, max_distance=MAX_CONNECTION_DISTANCE)
        if new_state is None:
            # Debugging statement:
            # print("[DEBUG] No valid extension found.")
            continue  # collision or no extension

        # Add to tree
        new_cost = costs[tuple(nearest)] + precompute_distance(nearest, new_state)
        kd_tree = add_node(new_state, nodes, kd_tree)
        parents[tuple(new_state)] = nearest
        costs[tuple(new_state)] = new_cost

        edge_ids[tuple(new_state)] = p.addUserDebugLine(
            [nearest[0], nearest[1], 0.1],
            [new_state[0], new_state[1], 0.1],
            [1, 0, 0],
            lineWidth=1.0
        )

        # -----------------------------
        # Rewire nearby nodes
        # -----------------------------
        for idx in kd_tree.query_ball_point(new_state[:2], RADIUS):
            near_node = nodes[idx]
            # Check collision from new_state to near_node
            if is_edge_collision_free(new_state, near_node, obstacles):
                potential_cost = new_cost + precompute_distance(new_state, near_node)
                if potential_cost < costs[tuple(near_node)]:
                    # Remove old debug line
                    if tuple(near_node) in edge_ids:
                        p.removeUserDebugItem(edge_ids[tuple(near_node)])
                    parents[tuple(near_node)] = new_state
                    costs[tuple(near_node)] = potential_cost
                    edge_ids[tuple(near_node)] = p.addUserDebugLine(
                        [new_state[0], new_state[1], 0.1],
                        [near_node[0], near_node[1], 0.1],
                        [0, 1, 0],
                        lineWidth=1.0
                    )

    # -----------------------------
    # Connect to Goal
    # -----------------------------
    for idx in kd_tree.query_ball_point(goal[:2], RADIUS):
        near_node = nodes[idx]
        if is_edge_collision_free(near_node, goal, obstacles):
            potential_cost = costs[tuple(near_node)] + precompute_distance(near_node, goal)
            if potential_cost < costs.get(goal_node, float('inf')):
                parents[goal_node] = near_node
                costs[goal_node] = potential_cost
                p.addUserDebugLine(
                    [near_node[0], near_node[1], 0.1],
                    [goal[0], goal[1], 0.1],
                    [0, 1, 1],
                    lineWidth=1.5
                )

    # -----------------------------
    # Build Final Path
    # -----------------------------
    if parents.get(goal_node) is not None:
        best_path = []
        cur = goal
        visited_nodes = set()
        while cur is not None:
            if tuple(cur) in visited_nodes:
                print(f"[DEBUG] Error: Cycle detected at node {cur}")
                break
            visited_nodes.add(tuple(cur))
            best_path.append(cur)
            cur = parents.get(tuple(cur))
        best_path.reverse()

        # Simplify path
        simplified_path = simplify_path(best_path, obstacles)
        total_cost = sum(precompute_distance(simplified_path[i], simplified_path[i + 1])
                         for i in range(len(simplified_path) - 1))
        print(f"Simplified path cost: {total_cost:.2f}")
        #best_path = simplified_path

    return best_path


# --------------------------
# Moving the Fire Truck
# --------------------------
def move_fire_truck_along_path(path, fire_truck):
    """Moves the fire truck in PyBullet simulation along the path."""
    for state in path:
        pos = [state[0], state[1], 0.1]
        theta = state[2]
        orientation = p.getQuaternionFromEuler([0, 0, theta])
        p.resetBasePositionAndOrientation(fire_truck, pos, orientation)
        p.stepSimulation()
        time.sleep(0.05)


# ----------------------
# Visualization (matplotlib)
# ----------------------
def visualize_path(path, obstacles, start, goal):
    """Visualizes the path, obstacles, start, and goal using matplotlib."""
    plt.figure(figsize=(10, 10))

    for obs_x, obs_y, _ in obstacles:
        square = plt.Rectangle(
            (obs_x - 0.5, obs_y - 0.5),
            1.0, 1.0,
            color='black',
            fill=True
        )
        plt.gca().add_patch(square)

    if path:
        path_x = [state[0] for state in path]
        path_y = [state[1] for state in path]
        plt.plot(path_x, path_y, c='red', linewidth=2, label="Path")
        plt.scatter(path_x, path_y, c='blue', s=50, label="Waypoints")
    else:
        print("[DEBUG] No path to visualize.")

    plt.scatter(start[0], start[1], c='green', s=200, label="Start", marker="o")
    plt.scatter(goal[0], goal[1], c='purple', s=200, label="Goal", marker="X")

    plt.xlim(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS)
    plt.ylim(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS)
    plt.axis('equal')
    plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.title("RRT* Path Planning (Bicycle Model)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# -----------
# Main Script
# -----------
if __name__ == "__main__":
    p.connect(p.GUI)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "..", "urdf")
    p.setAdditionalSearchPath(urdf_path)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)

    start = (-4, 2, 0)
    goal = (6, -3, 0)

    fire_truck = p.loadURDF("../urdf/fire_truck.urdf", [start[0], start[1], 0.1], [0, 0, 0, 1])

    obstacles = create_environment2(goal)
    print("Goal:", goal)

    path = rrt_star(start, goal, obstacles)

    if path:
        print("Path found!")
        np.savetxt("best_path.csv", path, delimiter=",")
        print(path)
        move_fire_truck_along_path(path, fire_truck)
        visualize_path(path, obstacles, start, goal)
    else:
        print("No path found.")

    p.disconnect()


