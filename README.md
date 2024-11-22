# PDM-group-33
firetruck_simulation/
├── urdf/
│   ├── firetruck.urdf       # Simplified fire truck model
│   ├── ladder.xacro         # Extendable ladder definition
│   └── wheel.xacro          # Modular wheel component
├── src/
│   ├── planner/
│   │   ├── planner.py       # Path planning algorithm
│   │   └── obstacle_avoidance.py
│   ├── control/
│       ├── drive_control.py # Fire truck movement control
│       └── ladder_control.py
├── launch/
│   ├── firetruck_world.launch.xml
│   └── gazebo.launch.py     # Starts the Gazebo environment
├── worlds/
│   ├── city_environment.world # Gazebo world file
├── README.md
└── requirements.txt          # Python dependencies
firetruck_simulation/
├── urdf/
│   ├── firetruck.urdf       # Simplified fire truck model
│   ├── ladder.xacro         # Extendable ladder definition
│   └── wheel.xacro          # Modular wheel component
├── src/
│   ├── planner/
│   │   ├── planner.py       # Path planning algorithm
│   │   └── obstacle_avoidance.py
│   ├── control/
│       ├── drive_control.py # Fire truck movement control
│       └── ladder_control.py
├── launch/
│   ├── firetruck_world.launch.xml
│   └── gazebo.launch.py     # Starts the Gazebo environment
├── worlds/
│   ├── city_environment.world # Gazebo world file
├── README.md
└── requirements.txt          # Python dependencies
