import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree

# Constants for the bicycle model
MAX_STEERING_ANGLE = np.pi / 4  # Maximum steering angle
MAX_VELOCITY = 1.0             # Maximum forward velocity
TURNING_RADIUS = 1.0           # Minimum turning radius
DT = 0.1                       # Time step for simulating motion

# RRT Parameters
MAX_NODES = 1000
GOAL_THRESHOLD = 0.5
MAX_CONNECTION_DISTANCE = 1.0  # Maximum step size for connecting nodes
RADIUS = MAX_CONNECTION_DISTANCE * 2
NUM_INITIAL_BRANCHES = 10      # Increased branching from the root
GOAL_BIAS = 0.0                # 10% of samples directed toward the goal
ENVIRONMENT_BOUNDS = 8        # Size of the environment (-10 to 10 in x/y)

ROBOT_LENGTH = 1.0  # Length of the robot
ROBOT_WIDTH = 0.5   # Width of the robot

# Define the environment
def create_environment(goal):
    """Creates a PyBullet environment with larger obstacles, borders, and a goal marker."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # Bigger obstacles


def create_environment2(goal):
    """Creates a PyBullet environment with larger obstacles, borders, and a goal marker."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # Bigger obstacles
    obstacles = [
        (2, 1, 0.5), (4, -2, 0.5), (-3, -3, 0.5),
        (1, -2, 0.5), (-4, 4, 0.5), (3, -4, 0.5),
        (5, 0, 0.5), (-5, -5, 0.5), (0, 5, 0.5)
    ]

    for x, y, z in obstacles:
        p.loadURDF("cube.urdf", [x, y, z], globalScaling=1.0)

    # Add borders as obstacles
    for x in range(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS + 1, 2):
        p.loadURDF("cube.urdf", [x, -ENVIRONMENT_BOUNDS, 0.5], globalScaling=1.0)
        p.loadURDF("cube.urdf", [x, ENVIRONMENT_BOUNDS, 0.5], globalScaling=1.0)

    for y in range(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS + 1, 2):
        p.loadURDF("cube.urdf", [-ENVIRONMENT_BOUNDS, y, 0.5], globalScaling=1.0)
        p.loadURDF("cube.urdf", [ENVIRONMENT_BOUNDS, y, 0.5], globalScaling=1.0)

    # Add the goal marker as a sphere
    p.loadURDF("sphere_small.urdf", [goal[0], goal[1], GOAL_THRESHOLD], globalScaling=GOAL_THRESHOLD * 2)

    return obstacles
    
def get_robot_corners(state):
    x, y, theta = state
    dx = ROBOT_LENGTH / 2
    dy = ROBOT_WIDTH / 2

    # Corners in local robot coordinates
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

# Bicycle model dynamics
def bicycle_step(state, velocity, steering_angle, dt=DT):
    x, y, theta = state
    theta += velocity * np.tan(steering_angle) * dt
    x += velocity * np.cos(theta) * dt
    y += velocity * np.sin(theta) * dt
    return x, y, theta

# Collision checking
def is_collision_free(state, obstacles):
    # Get the robot's polygon
    robot_polygon = Polygon(get_robot_corners(state))

    # Check collision with each obstacle (assumed as circles for simplicity)
    for obs_x, obs_y, _ in obstacles:
        obstacle_circle = Point(obs_x, obs_y).buffer(0.5)  # Obstacle radius = 0.5
        if robot_polygon.intersects(obstacle_circle):
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
    """Adds a new node to the KD-tree and updates structures."""
    nodes.append(new_state)
    kd_tree = cKDTree([node[:2] for node in nodes])  # Rebuild KD-tree with all nodes
    return kd_tree

# RRT implementation
def rrt_star(start, goal, obstacles):
    nodes = [start]
    parents = {tuple(start): None}
    costs = {tuple(start): 0}
    best_path = None
    best_cost = float('inf')
    edge_ids = {}  # To store PyBullet debug line IDs for edges

    # Create the initial KD-tree
    kd_tree = cKDTree([start[:2]])

    # Create initial branches in random directions
    for _ in range(NUM_INITIAL_BRANCHES):
        steering_angle = random.uniform(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
        initial_branch = bicycle_step(start, MAX_VELOCITY, steering_angle)
        if is_collision_free(initial_branch, obstacles):
            kd_tree = add_node(initial_branch, nodes, kd_tree)
            parents[tuple(initial_branch)] = start
            costs[tuple(initial_branch)] = costs[tuple(start)] + np.linalg.norm(
                np.array(start[:2]) - np.array(initial_branch[:2])
            )
            # Visualize the initial edges in red
            edge_id = p.addUserDebugLine(
                [start[0], start[1], 0.1],
                [initial_branch[0], initial_branch[1], 0.1],
                [1, 0, 0],  # Red
                lineWidth=1.0
            )
            edge_ids[tuple(initial_branch)] = edge_id

    for i in range(MAX_NODES):
        # Print progress every 100 nodes
        if i % 100 == 0:
            print(f"Processing node {i} / {MAX_NODES}")

        # Sample a random state
        if random.random() < GOAL_BIAS:
            rand_state = goal
        else:
            rand_x = random.uniform(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS)
            rand_y = random.uniform(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS)
            rand_theta = random.uniform(-np.pi, np.pi)
            rand_state = (rand_x, rand_y, rand_theta)

        # Find the nearest node using KD-tree
        _, nearest_idx = kd_tree.query(rand_state[:2])
        nearest = nodes[nearest_idx]

        # Move toward the random state with a limited connection distance
        direction = np.array(rand_state[:2]) - np.array(nearest[:2])
        distance = np.linalg.norm(direction)
        if distance > MAX_CONNECTION_DISTANCE:
            direction = direction / distance
            new_position = np.array(nearest[:2]) + direction * MAX_CONNECTION_DISTANCE
            new_state = (new_position[0], new_position[1], nearest[2])
        else:
            new_state = rand_state

        # Check for collisions
        if is_collision_free(new_state, obstacles):
            new_cost = costs[tuple(nearest)] + np.linalg.norm(
                np.array(nearest[:2]) - np.array(new_state[:2])
            )
            kd_tree = add_node(new_state, nodes, kd_tree)
            parents[tuple(new_state)] = nearest
            costs[tuple(new_state)] = new_cost

            # Remove the old edge and add the new edge
            if tuple(new_state) in edge_ids:
                p.removeUserDebugItem(edge_ids[tuple(new_state)])
            edge_id = p.addUserDebugLine(
                [nearest[0], nearest[1], 0.1],
                [new_state[0], new_state[1], 0.1],
                [1, 0, 0],  # Red
                lineWidth=1.0
            )
            edge_ids[tuple(new_state)] = edge_id

            # Rewiring nearby nodes
            nearby_indices = kd_tree.query_ball_point(new_state[:2], RADIUS)
            for idx in nearby_indices:
                near_node = nodes[idx]
                # Check if the edge between new_state and near_node is collision-free
                if is_edge_collision_free(new_state, near_node, obstacles):
                    potential_cost = new_cost + np.linalg.norm(
                        np.array(new_state[:2]) - np.array(near_node[:2])
                    )
                    if potential_cost < costs[tuple(near_node)]:
                        # Remove the old edge for near_node
                        old_edge = edge_ids.get(tuple(near_node))
                        if old_edge is not None:
                            p.removeUserDebugItem(old_edge)

                        # Update the parent and cost
                        parents[tuple(near_node)] = new_state
                        costs[tuple(near_node)] = potential_cost

                        
                        new_edge_id = p.addUserDebugLine(
                            [new_state[0], new_state[1], 0.1],
                            [near_node[0], near_node[1], 0.1],
                            [1, 0, 0], 
                            lineWidth=1.0
                        )
                        edge_ids[tuple(near_node)] = new_edge_id

    # Final Optimization: Find the best path to the goal
    for node in nodes:
        if np.linalg.norm(np.array(node[:2]) - np.array(goal[:2])) < GOAL_THRESHOLD:
            path_cost = costs[tuple(node)]
            if path_cost < best_cost:
                best_cost = path_cost
                best_path = []
                current = node
                while current is not None:
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
    p.setAdditionalSearchPath("/home/basil/PDM/Project/PDM-group-33/urdf")

    start = (0, 0, 0)
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
        # Move the fire truck along the path
        move_fire_truck_along_path(path, fire_truck)
        # Visualize the path
        visualize_path(path, obstacles, start, goal)
    else:
        print("No path found.")

    # Disconnect the PyBullet physics server
    p.disconnect()



# remove redundancy
# improve final plot
# sneller maken