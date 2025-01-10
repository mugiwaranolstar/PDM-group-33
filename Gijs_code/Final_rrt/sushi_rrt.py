import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import time
#zoeken tot pad is gevonden
# Constants for the bicycle model and environment
ENVIRONMENT_BOUNDS = 7
GOAL_THRESHOLD = 0.5
MAX_STEERING_ANGLE = np.pi / 4
MAX_VELOCITY = 1.2
DT = 0.1


def bicycle_step(state, velocity, steering_angle, dt=DT):
    """
    Bicycle model one step forward.
    state = (x, y, theta).
    """
    x, y, theta = state
    theta_new = theta + velocity * np.tan(steering_angle) * dt
    x_new = x + velocity * np.cos(theta_new) * dt
    y_new = y + velocity * np.sin(theta_new) * dt
    return (x_new, y_new, theta_new)


def create_environment2(goal):
    """
    Same environment creation as before.
    """
    if p.isConnected() == 0:
        p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    obstacles = [
        (2, 1, 0.5), (4, -2, 0.5), (-3, -3, 0.5),
        (1, -2, 0.5), (-4, 4, 0.5), (2, 0, 0.5),
        (5, 1, 0.5), (-5, -5, 0.5), (0, 5, 0.5),
        (0, -4, 0.5), (-4, 0, 0.5), (4, 4, 0.5),
        (-1, -1, 0.5), (0, 2, 0.5), (-1, -5, 0.5),
        (2, -2, 0.5), (0, 1.5, 0.5), (3, -5, 0.5),
        (2, -1.5, 0.5)
    ]
    for x, y, z in obstacles:
        p.loadURDF("cube.urdf", [x, y, z], globalScaling=1.0)

    # Borders
    for x in range(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS):
        p.loadURDF("cube.urdf", [x, -ENVIRONMENT_BOUNDS, 0.5], globalScaling=1.0)
        p.loadURDF("cube.urdf", [x, ENVIRONMENT_BOUNDS, 0.5], globalScaling=1.0)
    for y in range(-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS):
        p.loadURDF("cube.urdf", [-ENVIRONMENT_BOUNDS, y, 0.5], globalScaling=1.0)
        p.loadURDF("cube.urdf", [ENVIRONMENT_BOUNDS, y, 0.5], globalScaling=1.0)

    # Goal marker
    p.loadURDF(
        "sphere_small.urdf",
        [goal[0], goal[1], GOAL_THRESHOLD],
        globalScaling=GOAL_THRESHOLD * 2
    )
    return obstacles


def calculate_euclidean_distance(path):
    """
    Total distance ignoring theta.
    """
    total_distance = 0.0
    for i in range(len(path) - 1):
        (x1, y1, _), (x2, y2, _) = path[i], path[i + 1]
        total_distance += math.hypot(x2 - x1, y2 - y1)
    return total_distance


def calculate_steering_sum(path):
    """
    Estimate total steering change from heading differences.
    """
    total_steering = 0.0
    for i in range(len(path) - 2):
        _, _, theta1 = path[i]
        _, _, theta2 = path[i + 1]
        _, _, theta3 = path[i + 2]
        angle_diff1 = theta2 - theta1
        angle_diff2 = theta3 - theta2
        total_steering += abs(angle_diff2 - angle_diff1)
    return total_steering


class BicycleRRT:
    """
    RRT with bicycle kinematics and a final "best path" selection.
    """

    class Node:
        def __init__(self, x, y, theta):
            self.x = x
            self.y = y
            self.theta = theta
            # path of sub-steps
            self.path_x = []
            self.path_y = []
            self.path_theta = []
            self.parent = None

            # Track cost from the start (distance traveled along edges)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 robot_radius=0.2,
                 expand_dis=2.0,
                 path_resolution=0.2,
                 max_iter=1500,
                 goal_sample_rate=20):
        """
        start, goal = (x, y, theta)
        """
        self.start = self.Node(start[0], start[1], start[2])
        self.start.cost = 0.0
        self.end = self.Node(goal[0], goal[1], goal[2])

        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]

        self.robot_radius = robot_radius
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate

        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, animation=True):
        """
        Do all iterations, then pick best path to goal (if any).
        """
        self.node_list = [self.start]

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            if new_node is not None and self.check_collision(new_node):
                self.node_list.append(new_node)

            if animation and i % 25 == 0:
                self.draw_graph(rnd_node)

        # --------------------------------------------
        # After all iterations, pick the best path
        # --------------------------------------------
        best_goal_node = None
        best_cost = float('inf')

        # Try to connect each node to the goal
        for node in self.node_list:
            if self.distance(node, self.end) <= self.expand_dis:
                # Attempt final steer from node to goal
                final_node = self.steer(node, self.end, self.expand_dis)
                if final_node is not None and self.check_collision(final_node):
                    total_cost = node.cost + self.path_cost(final_node)
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_goal_node = final_node

        if best_goal_node is not None:
            return self.generate_final_course(best_goal_node)
        else:
            return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Use bicycle step increments from from_node to to_node.
        """
        new_node = self.Node(from_node.x, from_node.y, from_node.theta)

        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        desired_angle = math.atan2(dy, dx)

        angle_diff = desired_angle - from_node.theta
        # Normalize angle to [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        steering_angle = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, angle_diff))

        dist = math.hypot(dx, dy)
        if dist > extend_length:
            dist = extend_length

        n_expand = int(math.floor(dist / self.path_resolution))

        temp_state = (new_node.x, new_node.y, new_node.theta)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        new_node.path_theta = [new_node.theta]

        # Accumulate how much distance we move
        traveled_dist = 0.0

        for _ in range(n_expand):
            prev_x, prev_y, _ = temp_state
            temp_state = bicycle_step(temp_state, MAX_VELOCITY, steering_angle, DT)
            new_node.path_x.append(temp_state[0])
            new_node.path_y.append(temp_state[1])
            new_node.path_theta.append(temp_state[2])
            traveled_dist += math.hypot(temp_state[0] - prev_x, temp_state[1] - prev_y)

        leftover = dist - n_expand * self.path_resolution
        if leftover > 0.0:
            # smaller final step
            sub_steps = leftover / self.path_resolution
            prev_x, prev_y, _ = temp_state
            temp_state = bicycle_step(temp_state, MAX_VELOCITY, steering_angle, DT * sub_steps)
            new_node.path_x.append(temp_state[0])
            new_node.path_y.append(temp_state[1])
            new_node.path_theta.append(temp_state[2])
            traveled_dist += math.hypot(temp_state[0] - prev_x, temp_state[1] - prev_y)

        new_node.x = new_node.path_x[-1]
        new_node.y = new_node.path_y[-1]
        new_node.theta = new_node.path_theta[-1]
        new_node.parent = from_node

        # Update cost = parent's cost + traveled distance
        new_node.cost = from_node.cost + traveled_dist

        return new_node

    def generate_final_course(self, goal_node):
        """
        Reconstruct path from goal_node to start by walking parents.
        Returns list of (x, y, theta).
        """
        path = []
        node = goal_node
        while node is not None:
            # add sub-steps in reverse
            for i in range(len(node.path_x) - 1, -1, -1):
                path.append((node.path_x[i], node.path_y[i], node.path_theta[i]))
            node = node.parent
        path.reverse()
        return path

    def get_random_node(self):
        """
        Random or goal-biased sample.
        """
        if random.randint(0, 100) > self.goal_sample_rate:
            x = random.uniform(self.min_rand, self.max_rand)
            y = random.uniform(self.min_rand, self.max_rand)
            theta = random.uniform(-math.pi, math.pi)
            return self.Node(x, y, theta)
        else:
            return self.Node(self.end.x, self.end.y, self.end.theta)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        """
        Return index of the node in node_list closest (XY distance) to rnd_node.
        """
        dlist = [
            (node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
            for node in node_list
        ]
        min_index = dlist.index(min(dlist))
        return min_index

    def check_collision(self, node):
        """
        Check if path in 'node' is collision-free.
        """
        if node is None:
            return False

        for x, y in zip(node.path_x, node.path_y):
            # environment bounds
            if not (-ENVIRONMENT_BOUNDS <= x <= ENVIRONMENT_BOUNDS and
                    -ENVIRONMENT_BOUNDS <= y <= ENVIRONMENT_BOUNDS):
                return False
            # obstacles
            for (ox, oy, size) in self.obstacle_list:
                dx = ox - x
                dy = oy - y
                if (dx**2 + dy**2) <= (size + self.robot_radius)**2:
                    return False
        return True

    def path_cost(self, node):
        """
        Compute the extra distance traveled in node's path_x, path_y
        from its parent. We can also just return node.cost - node.parent.cost,
        but let's be explicit.
        """
        if node is None or node.parent is None:
            return 0.0
        dist = 0.0
        px = node.path_x
        py = node.path_y
        for i in range(len(px) - 1):
            dist += math.hypot(px[i+1] - px[i], py[i+1] - py[i])
        return dist

    @staticmethod
    def distance(node1, node2):
        """
        XY distance between two nodes.
        """
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    def draw_graph(self, rnd=None):
        """
        Plot the RRT tree.
        """
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent is not None:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x,   self.end.y,   "xr")
        plt.axis("equal")
        plt.xlim(self.min_rand - 1, self.max_rand + 1)
        plt.ylim(self.min_rand - 1, self.max_rand + 1)
        plt.grid(True)
        plt.pause(0.001)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(math.radians(d)) for d in deg]
        yl = [y + size * math.sin(math.radians(d)) for d in deg]
        plt.plot(xl, yl, color)


def main():
    print("RRT with bicycle model -- collecting all nodes, then picking best path at the end.")

    p.connect(p.GUI)

    start = (-4, 2, 0.0)
    goal  = (6, -3, 0.0)

    obstacles = create_environment2(goal)
    obstacle_list = [(x, y, 0.5) for x, y, _ in obstacles]

    robot_length = 1.0
    robot_width  = 0.5
    robot_radius = (robot_length**2 + robot_width**2)**0.5 / 2.0

    rrt = BicycleRRT(
        start=start,
        goal=goal,
        obstacle_list=obstacle_list,
        rand_area=[-ENVIRONMENT_BOUNDS, ENVIRONMENT_BOUNDS],
        robot_radius=robot_radius,
        path_resolution=0.2,
        expand_dis=2.0,
        max_iter=1500,
        goal_sample_rate=20
    )

    path = rrt.planning(animation=True)

    if path is None:
        print("No path found.")
    else:
        print("Path found!")
        print("Number of waypoints in final path:", len(path))

        total_distance = calculate_euclidean_distance(path)
        total_steering = calculate_steering_sum(path)

        print(f"Total Euclidean Distance: {total_distance:.2f}")
        print(f"Total Steering Sum:       {total_steering:.2f}")
        np.savetxt("best_path_voorbeeld_rrt.csv", path, delimiter=",")
        # final visualization
        rrt.draw_graph()
        plt.plot(
            [s[0] for s in path],
            [s[1] for s in path],
            '-r', linewidth=2
        )
        plt.show()

    p.disconnect()


if __name__ == "__main__":
    main()
