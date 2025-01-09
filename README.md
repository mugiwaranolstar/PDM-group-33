# PDM-group-33

# Fire Truck Simulation with Planner Integration 🚒

This project simulates a simplified fire truck robot in a virtual environment. It combines a custom **URDF model** of the fire truck with a **path-planning algorithm** that enables the robot to navigate through obstacles to reach a target building.

---

## Overview
The project demonstrates the integration of:
- A **simplified fire truck URDF model**, showcasing the vehicle's key features such as wheels and a chassis.
- A **custom planner** that allows the fire truck to:
  1. Navigate through an environment with static obstacles.
  2. Reach a designated target (e.g., a building).
  3. Adapt to various environments based on the vehicle's kinematic constraints

This simulation is ideal for testing robotic control systems and planners in a rescue scenario.

---

## Features
- **Simplified Fire Truck URDF Model**:
  - Includes basic geometry for wheels and a chassis.
  - Configurable for different scenarios and dimensions.
  
- **Path Planning**:
  - Uses RRT* to find the optimal path from start to finish avoiding while static obstacles.
  - Utilizes the MPC method to follow the path.

---

### Installation
1. Install the simulation environment:

``` {.sourceCode .bash}
pip3 install urdfenvs
```

2. Clone the repository:
   ```bash
   git clone git@github.com:mugiwaranolstar/PDM-group-33.git
   ```

3. Go to the scripts' repository:
   ```bash
   cd PDM-group-33/scripts
   ```

4. Run the full code:
   ```bash
   python3 launch_code.py
   ```

This runs the three scripts:
**rrt_star33_final.py**, **mpc_constraints.py** and **simulation.py** in a row.
If desired, the three scripts can be run separately:
1. To run the global planner:
   ```bash
   python3 rrt_star33_final.py
   ```

2. To run the local planner:
   ```bash
   python3 mpc_constraints.py
   ```

3. To simulate the movement of the robot in the environment:
   ```bash
   python3 simulation.py
   ```

Note that the three scripts need to be run in the aforementioned order for them to work together.