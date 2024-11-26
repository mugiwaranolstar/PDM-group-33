import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
from heapq import heappush, heappop

# Constants
MAX_STEERING_ANGLE = np.pi / 3  # Increased steering angle for sharper turns
MAX_VELOCITY = 2.5             # Increased forward velocity for larger steps
TURNING_RADIUS = 1.0           # Minimum turning radius
DT = 0.2                       # Larger time step for bigger movements
GOAL_THRESHOLD = 0.65           # Threshold for goal proximity
ENVIRONMENT_BOUNDS = 10        # Environment bounds (-10 to 10 in x/y)
WHEELBASE = 0.87

# Motion Primitive Parameters
MOTION_PRIMITIVES = [
    (MAX_VELOCITY, 0.0),               # Straight
    (MAX_VELOCITY, np.pi / 6),         # Slight right
    (MAX_VELOCITY, -np.pi / 6),        # Slight left
    (MAX_VELOCITY, np.pi / 4),         # Moderate right
    (MAX_VELOCITY, -np.pi / 4),        # Moderate left
]

# Define the environment (from old code)
def create_environment(goal):
    """Creates a PyBullet environment with larger obstacles, borders, and a goal marker."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # Obstacles
    obstacles = [
    (2, 1, 0.5), (-4, 4, 0.5), (0, 5, 0.5)  # Remove some obstacles
]

    for x, y, z in obstacles:
        p.loadURDF("cube.urdf", [x, y, z], globalScaling=1.0, useFixedBase=True)

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

# Bicycle model dynamics for firetruck
def bicycle_step(state, velocity, steering_angle, dt=DT):
    x, y, theta = state
    beta = np.arctan(0.5 * np.tan(steering_angle))  # Adjusted for pivot at the front wheel
    x += velocity * np.cos(theta + beta) * dt
    y += velocity * np.sin(theta + beta) * dt
    theta += velocity / WHEELBASE * np.tan(steering_angle) * dt
    return x, y, theta

# Collision checking with C-Space
def is_collision_free(state, obstacles, fire_truck_length=1.1, fire_truck_width=0.62):
    x, y, theta = state
    corners = np.array([
        [-fire_truck_length / 2, -fire_truck_width / 2],
        [fire_truck_length / 2, -fire_truck_width / 2],
        [fire_truck_length / 2, fire_truck_width / 2],
        [-fire_truck_length / 2, fire_truck_width / 2]
    ])

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    transformed_corners = np.dot(corners, rotation_matrix.T) + np.array([x, y])

    for obs_x, obs_y, _ in obstacles:
        if any(
            obs_x - 1.0 <= corner[0] <= obs_x + 1.0 and
            obs_y - 1.0 <= corner[1] <= obs_y + 1.0
            for corner in transformed_corners
        ):
            print(f"Collision detected at state: {state} with obstacle: {obs_x, obs_y}")
            return False

    if not all(-ENVIRONMENT_BOUNDS <= corner[0] <= ENVIRONMENT_BOUNDS and
               -ENVIRONMENT_BOUNDS <= corner[1] <= ENVIRONMENT_BOUNDS for corner in transformed_corners):
        print(f"Out of bounds detected at state: {state}")
        return False

    return True


# Motion Primitives Planning with A* and Visualization
def plan_motion_primitives(start, goal, obstacles):
    open_list = []
    heappush(open_list, (0, start))
    visited = set()
    parents = {tuple(start): None}
    g_costs = {tuple(start): 0}
    closest_distance = float('inf')
    nodes_expanded = 0  # Count expanded nodes

    while open_list:
        _, current = heappop(open_list)
        visited.add(tuple(current))
        nodes_expanded += 1  # Increment node counter

        distance_to_goal = np.linalg.norm(np.array(current[:2]) - np.array(goal[:2]))
        if distance_to_goal < closest_distance:
            closest_distance = distance_to_goal
            print(f"Closest distance to goal: {closest_distance:.2f}")

        if distance_to_goal < GOAL_THRESHOLD:
            print(f"Path found after expanding {nodes_expanded} nodes!")
            path = []
            while current is not None:
                path.append(current)
                current = parents[tuple(current)]
            return path[::-1]

        for velocity, steering_angle in MOTION_PRIMITIVES:
            new_state = bicycle_step(current, velocity, steering_angle)
            state_tuple = tuple(new_state)

            # Visualize all explored paths
            start_pos = [current[0], current[1], 0.1]
            end_pos = [new_state[0], new_state[1], 0.1]
            p.addUserDebugLine(start_pos, end_pos, [1, 1, 0], lineWidth=0.5)  # Yellow lines for all paths
            p.stepSimulation()  # Ensure visualization updates in real-time

            if state_tuple in visited or not is_collision_free(new_state, obstacles):
                continue

            g_cost = g_costs[tuple(current)] + velocity * DT
            h_cost = np.linalg.norm(np.array(new_state[:2]) - np.array(goal[:2]))
            total_cost = g_cost + h_cost

            if state_tuple not in g_costs or g_cost < g_costs[state_tuple]:
                g_costs[state_tuple] = g_cost
                parents[state_tuple] = current
                heappush(open_list, (total_cost, new_state))

                # Visualize valid paths in green
                p.addUserDebugLine(start_pos, end_pos, [0, 1, 0], lineWidth=1.0)

    print(f"No path found after expanding {nodes_expanded} nodes.")
    return None



# Visualization using Matplotlib with Firetruck Footprint
def visualize_path(path, obstacles, start, goal, fire_truck_length=2.0, fire_truck_width=1.0):
    """
    Visualizes the path, obstacles, firetruck footprint, and collision space (C-space).
    """
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Draw obstacles (1:1 size)
    for obs_x, obs_y, _ in obstacles:
        # Draw the actual obstacle
        obstacle_rect = plt.Rectangle((obs_x - 0.5, obs_y - 0.5), 1.0, 1.0, color='black', fill=True, label="Obstacle")
        ax.add_patch(obstacle_rect)

        # Draw the C-space (expanded area around the obstacle)
        cspace_margin_x = fire_truck_length / 2
        cspace_margin_y = fire_truck_width / 2
        cspace_rect = plt.Rectangle(
            (obs_x - 0.5 - cspace_margin_x, obs_y - 0.5 - cspace_margin_y),
            1.0 + 2 * cspace_margin_x,
            1.0 + 2 * cspace_margin_y,
            color='red',
            alpha=0.3,  # Make it semi-transparent
            label="Collision Space"
        )
        ax.add_patch(cspace_rect)

    # Draw the path with firetruck footprints
    if path:
        for state in path:
            x, y, theta = state
            corners = np.array([
                [-fire_truck_length / 2, -fire_truck_width / 2],
                [fire_truck_length / 2, -fire_truck_width / 2],
                [fire_truck_length / 2, fire_truck_width / 2],
                [-fire_truck_length / 2, fire_truck_width / 2]
            ])
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            transformed_corners = np.dot(corners, rotation_matrix.T) + np.array([x, y])
            polygon = plt.Polygon(transformed_corners, closed=True, color='blue', alpha=0.3)
            ax.add_patch(polygon)

        path_x = [state[0] for state in path]
        path_y = [state[1] for state in path]
        plt.plot(path_x, path_y, c='red', linewidth=2, label="Path")
        plt.scatter(path_x, path_y, c='blue', s=50, label="Waypoints")

    # Draw start and goal
    plt.scatter(start[0], start[1], c='green', s=200, label="Start", marker="o")
    plt.scatter(goal[0], goal[1], c='purple', s=200, label="Goal", marker="X")

    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.title("Motion Primitives Path Planning with Collision Space")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()



# Moving the firetruck along the path
def move_fire_truck_along_path(path, fire_truck):
    """
    Moves the firetruck along the path, updating the front wheel's rotation based on the steering angle.
    """
    for i in range(1, len(path)):
        prev_state = path[i - 1]
        current_state = path[i]

        # Compute movement details
        dx = current_state[0] - prev_state[0]
        dy = current_state[1] - prev_state[1]
        distance = np.sqrt(dx**2 + dy**2)
        steering_angle = np.arctan2(dy, dx) - prev_state[2]  # Change in orientation

        # Update firetruck position and orientation
        pos = [current_state[0], current_state[1], 0.1]
        theta = current_state[2]
        orientation = p.getQuaternionFromEuler([0, 0, theta])
        p.resetBasePositionAndOrientation(fire_truck, pos, orientation)

        # Update the front wheel's steering angle
        front_wheel_joint = 0  # Index of the front_wheel_joint in PyBullet
        p.resetJointState(fire_truck, front_wheel_joint, targetValue=steering_angle)

        # Step simulation
        p.stepSimulation()
        time.sleep(0.05)


def print_path_steps(path):
    """
    Prints the steps the firetruck takes to follow the path.
    Each step includes the distance driven and the turn angle.
    """
    if not path or len(path) < 2:
        print("Path is too short to analyze steps.")
        return

    print("\nSteps to reach the goal:")
    for i in range(1, len(path)):
        prev_state = path[i - 1]
        current_state = path[i]

        # Calculate distance moved
        dx = current_state[0] - prev_state[0]
        dy = current_state[1] - prev_state[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Calculate turn angle
        dtheta = current_state[2] - prev_state[2]
        turn_angle = np.degrees(dtheta)  # Convert to degrees

        # Determine movement type
        if np.abs(turn_angle) > 1e-2:  # Significant turn
            print(f"Turn {turn_angle:.2f} degrees.")
        if distance > 1e-2:  # Significant forward movement
            print(f"Drive forward {distance:.2f} units.")


def test_front_wheel_steering(fire_truck, front_wheel_joint, num_cycles=2, max_steering_angle=np.pi/4, step_size=0.05):
    """
    Gradually rotate the front wheel left and right in place.

    Parameters:
    - fire_truck: The PyBullet body ID of the firetruck.
    - front_wheel_joint: The joint index of the front wheel in the URDF.
    - num_cycles: Number of left-right cycles to perform.
    - max_steering_angle: Maximum steering angle for the test.
    - step_size: Incremental step size for the steering angle.
    """
    for _ in range(num_cycles):
        # Gradually turn the wheel to the left
        angle = 0
        while angle < max_steering_angle:
            angle += step_size
            p.resetJointState(fire_truck, front_wheel_joint, targetValue=min(angle, max_steering_angle))
            p.stepSimulation()
            time.sleep(0.1)

        # Gradually turn the wheel to the right
        angle = max_steering_angle
        while angle > -max_steering_angle:
            angle -= step_size
            p.resetJointState(fire_truck, front_wheel_joint, targetValue=max(angle, -max_steering_angle))
            p.stepSimulation()
            time.sleep(0.1)

        # Gradually return to neutral
        angle = -max_steering_angle
        while angle < 0:
            angle += step_size
            p.resetJointState(fire_truck, front_wheel_joint, targetValue=min(angle, 0))
            p.stepSimulation()
            time.sleep(0.1)

if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath("/home/gijs/gym_envs_urdf/PDM_project/PDM-group-33/urdf")

    start = (0, 0, 0)
    goal = (4, 2, 0)

    # Load the firetruck URDF
    fire_truck = p.loadURDF("fire_truck.urdf", [start[0], start[1], 0.1], [0, 0, 0, 1])

    # Get the index of the front wheel joint
    front_wheel_joint = 0  # Confirm index from URDF structure

    # Test front wheel steering
    #print("Testing front wheel steering...")
    #test_front_wheel_steering(fire_truck, front_wheel_joint)

    # Create the environment
    obstacles = create_environment(goal)

    print("Planning motion primitives...")
    path = plan_motion_primitives(start, goal, obstacles)

    if path:
        print("Path found!")
        print_path_steps(path)  # Output the steps to the terminal
        move_fire_truck_along_path(path, fire_truck)
        visualize_path(path, obstacles, start, goal)
    else:
        print("No path found.")
    time.sleep(10)
    p.disconnect()

