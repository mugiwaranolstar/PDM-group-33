# PDM-group-33

# Fire Truck Simulation with Planner Integration 🚒

This project simulates a simplified fire truck robot in a virtual environment. It combines a custom **URDF model** of the fire truck with a **path-planning algorithm** that enables the robot to navigate through obstacles, reach a target building, and raise its ladder for fire rescue missions.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Future Improvements](#future-improvements)
7. [Acknowledgments](#acknowledgments)

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
  - Utilizes a goal-driven approach to navigate toward the building.
  
- **Ladder Deployment**:
  - Simulates the ladder raising once the fire truck reaches the target.

---

## Getting Started

### Prerequisites
- [ROS 2](https://docs.ros.org/en/rolling/Installation.html)
- [Gazebo](http://gazebosim.org/)
- Python 3.9+ (for the planner scripts)
- Dependencies for the planner:
  - `numpy`
  - `scipy`
  - `matplotlib`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/firetruck_simulation.git
   cd firetruck_simulation
