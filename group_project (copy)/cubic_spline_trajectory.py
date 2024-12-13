import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Constants for the environment
ENVIRONMENT_BOUNDS = 8

start = [-4,2]
end = [6,-3.5]

# Waypoints approximated from the image
waypoints = [
    start,
    [-3.2, 1.5],
    [-3.5, 1.4],
    [-1.8, 1.8],
    [-1.8, 1.9],
    [0.2, 1.7],
    [0.5, 1.4],
    [2, -1],
    [2, -2.5],
    [3.5, -3],
    end
]

# Extract x and y coordinates
waypoints = np.array(waypoints)
x = waypoints[:, 0]
y = waypoints[:, 1]

# Apply cubic spline interpolation
cs_x = CubicSpline(range(len(x)), x)  # Interpolate x as a function of index
cs_y = CubicSpline(range(len(y)), y)  # Interpolate y as a function of index

# Generate smooth trajectory
samples = np.linspace(0, len(x) - 1, 100)  # Adjust sampling points
smooth_x = cs_x(samples)
smooth_y = cs_y(samples)

# Define obstacles for the plot
obstacles = [
    (2, 1), (4, -2), (-3, -3),
    (1, -2), (-4, 4), (3, -4),
    (5, 0), (-5, -5), (0, 5),
    (0, -2), (-4, 0), (4, 4),
    (-1, -1)
]

# Plot the environment
plt.figure(figsize=(8, 6))
plt.xlim(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS)
plt.ylim(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS)

# Plot obstacles as black squares
for obs in obstacles:
    plt.gca().add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, color='black'))

# Plot waypoints and smooth path
plt.plot(x, y, 'o', label="Waypoints")  # Original waypoints
plt.plot(smooth_x, smooth_y, '-', label="Cubic Spline Path")  # Smooth path
plt.scatter(waypoints[0, 0], waypoints[0, 1], color='green', label='Start', s=100)
plt.scatter(waypoints[-1, 0], waypoints[-1, 1], color='purple', label='Goal', s=100)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.title("Cubic Spline Path with Obstacles")
plt.show()
