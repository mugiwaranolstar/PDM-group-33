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
MAX_NODES = 900
GOAL_THRESHOLD = 0.5
MAX_CONNECTION_DISTANCE = 1.0
RADIUS = MAX_CONNECTION_DISTANCE * 2
NUM_INITIAL_BRANCHES = 10
GOAL_BIAS = 0.0
ENVIRONMENT_BOUNDS = 8

ROBOT_LENGTH = 1.0
ROBOT_WIDTH = 0.5

# Define the environment
def create_environment2(goal):
    """Creates a PyBullet environment with larger obstacles, borders, and a goal marker."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    obstacles = [
        (2, 1, 0.5), (4, -2, 0.5), (-3, -3, 0.5),
        (1, -2, 0.5), (-4, 4, 0.5), (3, -4, 0.5),
        (5, 0, 0.5), (-5, -5, 0.5), (0, 5, 0.5), 
        (0, -2, 0.5), (-4, 0, 0.5), (4, 4, 0.5),
        (-1, -1, 0.5)
    ]
    for x, y, z in obstacles:
        p.loadURDF("cube.urdf", [x, y, z], globalScaling=1.0)

    for x in range(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS + 1, 2):
        p.loadURDF("cube.urdf", [x, -ENVIRONMENT_BOUNDS, 0.5], globalScaling=1.0)
        p.loadURDF("cube.urdf", [x, ENVIRONMENT_BOUNDS, 0.5], globalScaling=1.0)

    for y in range(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS + 1, 2):
        p.loadURDF("cube.urdf", [-ENVIRONMENT_BOUNDS, y, 0.5], globalScaling=1.0)
        p.loadURDF("cube.urdf", [ENVIRONMENT_BOUNDS, y, 0.5], globalScaling=1.0)

    p.loadURDF("sphere_small.urdf", [goal[0], goal[1], GOAL_THRESHOLD], globalScaling=GOAL_THRESHOLD * 2)
    return obstacles

# Precompute Euclidean distances
def compute_distance(node1, node2):
    return np.linalg.norm(np.array(node1[:2]) - np.array(node2[:2]))

def bicycle_step(state, velocity, steering_angle, dt=DT):
    x, y, theta = state
    theta += velocity * np.tan(steering_angle) * dt
    x += velocity * np.cos(theta) * dt
    y += velocity * np.sin(theta) * dt
    return x, y, theta

def get_robot_corners(state):
    x, y, theta = state
    dx = ROBOT_LENGTH / 2
    dy = ROBOT_WIDTH / 2

    # Corners in the local robot coordinates
    local_corners = np.array([
        [-dx, -dy],
        [-dx, dy],
        [dx, dy],
        [dx, -dy]
    ])

    # Rotation matrix for the robot's orientation
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Transform corners to world coordinates
    world_corners = np.dot(local_corners, rotation_matrix.T) + np.array([x, y])
    return world_corners

def is_collision_free(state, obstacles):
    robot_corners = get_robot_corners(state)
    robot_polygon = Polygon(robot_corners)
    for obs_x, obs_y, _ in obstacles:
        if robot_polygon.intersects(Point(obs_x, obs_y).buffer(0.5)):
            return False
    return True

def is_edge_collision_free(start, end, obstacles, steps=5):
    for i in range(steps + 1):
        alpha = i / steps
        x = (1 - alpha) * start[0] + alpha * end[0]
        y = (1 - alpha) * start[1] + alpha * end[1]
        theta = (1 - alpha) * start[2] + alpha * end[2]
        intermediate_state = (x, y, theta)
        if not is_collision_free(intermediate_state, obstacles):
            return False
    return True

def add_node(new_state, nodes, kd_tree):
    nodes.append(new_state)
    # Efficiently update KD-tree
    kd_tree = cKDTree([node[:2] for node in nodes])  # Only rebuild when necessary
    return kd_tree

def rrt_star(start, goal, obstacles):
    nodes = [start]
    parents = {tuple(start): None}
    costs = {tuple(start): 0}
    distances = {}  # Precomputed distances
    best_path = None
    best_cost = float('inf')
    edge_ids = {}

    kd_tree = cKDTree([start[:2]])

    def precompute_distance(node1, node2):
        return compute_distance(node1, node2)

    for _ in range(NUM_INITIAL_BRANCHES):
        steering_angle = random.uniform(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
        branch = bicycle_step(start, MAX_VELOCITY, steering_angle)
        if is_collision_free(branch, obstacles):
            kd_tree = add_node(branch, nodes, kd_tree)
            parents[tuple(branch)] = start
            distances[(tuple(branch), tuple(start))] = precompute_distance(branch, start)
            costs[tuple(branch)] = distances[(tuple(branch), tuple(start))]
            edge_ids[tuple(branch)] = p.addUserDebugLine(
                [start[0], start[1], 0.1], [branch[0], branch[1], 0.1], [1, 0, 0], lineWidth=1.0)

    for i in range(MAX_NODES):
        if i % 100 == 0:
            print(f"Processing node {i} / {MAX_NODES}")
        rand_state = goal if random.random() < GOAL_BIAS else (
            random.uniform(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS),
            random.uniform(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS),
            random.uniform(-np.pi, np.pi),
        )
        _, nearest_idx = kd_tree.query(rand_state[:2])
        nearest = nodes[nearest_idx]

        direction = np.array(rand_state[:2]) - np.array(nearest[:2])
        distance = np.linalg.norm(direction)
        if distance > MAX_CONNECTION_DISTANCE:
            direction /= distance
            new_position = np.array(nearest[:2]) + direction * MAX_CONNECTION_DISTANCE
            new_state = (new_position[0], new_position[1], nearest[2])
        else:
            new_state = rand_state

        if is_collision_free(new_state, obstacles):
            new_cost = costs[tuple(nearest)] + precompute_distance(nearest, new_state)
            kd_tree = add_node(new_state, nodes, kd_tree)
            parents[tuple(new_state)] = nearest
            costs[tuple(new_state)] = new_cost

            if tuple(new_state) in edge_ids:
                p.removeUserDebugItem(edge_ids[tuple(new_state)])
            edge_ids[tuple(new_state)] = p.addUserDebugLine(
                [nearest[0], nearest[1], 0.1], [new_state[0], new_state[1], 0.1], [1, 0, 0], lineWidth=1.0)

            for idx in kd_tree.query_ball_point(new_state[:2], RADIUS):
                near_node = nodes[idx]
                if is_edge_collision_free(new_state, near_node, obstacles):
                    potential_cost = new_cost + precompute_distance(new_state, near_node)
                    if potential_cost < costs[tuple(near_node)]:
                        if tuple(near_node) in edge_ids:
                            p.removeUserDebugItem(edge_ids[tuple(near_node)])
                        parents[tuple(near_node)] = new_state
                        costs[tuple(near_node)] = potential_cost
                        edge_ids[tuple(near_node)] = p.addUserDebugLine(
                            [new_state[0], new_state[1], 0.1], [near_node[0], near_node[1], 0.1], [1, 0, 0], lineWidth=1.0)

    for node in nodes:
        if np.linalg.norm(np.array(node[:2]) - np.array(goal[:2])) < GOAL_THRESHOLD:
            if costs[tuple(node)] < best_cost:
                best_cost = costs[tuple(node)]
                best_path = []
                current = node
                while current:
                    best_path.append(current)
                    current = parents[tuple(current)]

    return best_path[::-1] if best_path else None





# Moving the fire truck along the path
def move_fire_truck_along_path(path, fire_truck):
    for state in path:
        pos = [state[0], state[1], 0.1]  # x, y, z
        theta = state[2]
        orientation = p.getQuaternionFromEuler([0, 0, theta])
        p.resetBasePositionAndOrientation(fire_truck, pos, orientation)
        p.stepSimulation()
        time.sleep(0.05)  # Small delay for visualization

# Visualization
def visualize_path(path, obstacles, start, goal):
    """Visualizes the path, obstacles, start, and goal using matplotlib."""
    plt.figure(figsize=(10, 10))

    # Plot obstacles as squares (to match PyBullet visualization)
    for obs_x, obs_y, _ in obstacles:
        square = plt.Rectangle((obs_x - 0.5, obs_y - 0.5), 1.0, 1.0, color='black', fill=True)  # Exact size match
        plt.gca().add_patch(square)

    # Plot path
    if path:
        path_x = [state[0] for state in path]
        path_y = [state[1] for state in path]
        plt.plot(path_x, path_y, c='red', linewidth=2, label="Path")
        plt.scatter(path_x, path_y, c='blue', s=50, label="Waypoints")
    else:
        print("No path to visualize!")

    # Mark start and goal positions
    plt.scatter(start[0], start[1], c='green', s=200, label="Start", marker="o")
    plt.scatter(goal[0], goal[1], c='purple', s=200, label="Goal", marker="X")

    # Configure plot
    plt.xlim(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS)
    plt.ylim(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS)
    plt.axis('equal')  # Ensure 1:1 aspect ratio
    plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.title("RRT Path Planning")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# Main
if __name__ == "__main__":
    # Initialize the PyBullet physics server
    p.connect(p.GUI)  # Connect only once
    # p.setAdditionalSearchPath("/home/basil/PDM/Project/PDM-group-33/urdf")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "..", "urdf")
    p.setAdditionalSearchPath(urdf_path)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)

    start = (-4, 2, 0)
    goal = (6, -3, 0)

    # Load the fire truck
    fire_truck = p.loadURDF("../urdf/fire_truck.urdf", [start[0], start[1], 0.1], [0, 0, 0, 1])

    # Create the environment with the goal marker
    obstacles = create_environment2(goal)

    print("Goal:", goal)
    # Plan the path
    path = rrt_star(start, goal, obstacles)

    if path:
        print("Path found!")
        np.savetxt("best_path.csv", path, delimiter=",")
        print(path)
        # Move the fire truck along the path
        move_fire_truck_along_path(path, fire_truck)
        # Visualize the path
        visualize_path(path, obstacles, start, goal)
        # to download in local_planner:
        # best_path_array = np.loadtxt("best_path.csv", delimiter=",")
    else:
        print("No path found.")

    # Disconnect the PyBullet physics server
    p.disconnect()



# remove redundancy
# improve final plot
# sneller maken