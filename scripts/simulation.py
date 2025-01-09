import os
import pandas as pd
import pybullet as p
import pybullet_data
import math
import numpy as np


ENVIRONMENT_BOUNDS = 7
GOAL_THRESHOLD = 0.5


script_dir = os.path.dirname(os.path.abspath(__file__))

# Import path
mpc_path = os.path.join(script_dir, "..", "csv_files", "MPC_path.csv")
df = pd.read_csv(mpc_path)
columns = ["x", "y", "yaw", "v"]
path = df[columns].to_numpy()

# Import Obstacles
obstacles_path = os.path.join(script_dir, "..", "csv_files", "obstacles.csv")
obstacles = np.array(pd.read_csv(obstacles_path))

# --------------------
# Environment Creation
# --------------------
def create_environment2(goal):
    """Creates a PyBullet environment with larger obstacles, borders, and a goal marker."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

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


def move_fire_truck_along_path(path, fire_truck):
    """Moves the fire truck in PyBullet simulation along the path with speed."""
    for state in path:
        pos = [state[0], state[1], 0.1] 
        theta = state[2]  
        speed = state[3]  
        
        velocity = [speed * math.cos(theta), speed * math.sin(theta), 0]

        orientation = p.getQuaternionFromEuler([0, 0, theta])
        p.resetBasePositionAndOrientation(fire_truck, pos, orientation)

        p.resetBaseVelocity(fire_truck, linearVelocity=velocity)

        p.stepSimulation()
    exit()


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

    move_fire_truck_along_path(path, fire_truck)

    p.disconnect()
    
