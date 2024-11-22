# PDM-group-33

# Fire Truck Simulation with Planner Integration 🚒

This project simulates a simplified fire truck robot in a virtual environment. It combines a custom **URDF model** of the fire truck with a **path-planning algorithm** that enables the robot to navigate through obstacles, reach a target building, and raise its ladder for fire rescue missions.

---

## Overview
The project demonstrates the integration of:
- A **simplified fire truck URDF model**, showcasing the vehicle's key features such as wheels, chassis, and extendable ladder.
- A **custom planner** that allows the fire truck to:
  1. Navigate through an environment with static obstacles.
  2. Reach a designated target (e.g., a building).
  3. Deploy and raise its ladder upon arrival.

This simulation is ideal for testing robotic control systems and planners in a rescue scenario.

---

## Features
- **Simplified Fire Truck URDF Model**:
  - Includes basic geometry for wheels, chassis, and an extendable ladder.
  - Configurable for different scenarios and dimensions.
  
- **Path Planning**:
  - Avoids static obstacles using custom algorithms.
  - Utilizes the motion primitives method to navigate toward the building.
  
- **Ladder Deployment**:
  - Uses an RRT* planner to find a way to get the top of the ladder to the target
  - Simulates the ladder raising once the fire truck reaches the building.

---

### Installation
1. Install the simulation environment:

``` {.sourceCode .bash}
pip3 install urdfenvs
```

2. Clone the repository:
   ```bash
   git clone git@github.com:mugiwaranolstar/PDM-group-33.git
   cd PDM-group-33

