import numpy as np
from heapdict import heapdict
import math
import dubins
import time
from utils.path_utils import calculate_path_length, angle_wrap
from car_model import CarModel
import utils.reeds_shepp as rs_curves
from utils.cubic_spline import calc_spline_course


# hybrid A star node
class Node:
    def __init__(self, grid_index, traj, curvature, cost, direction, parent_index):
        self.grid_index = grid_index  # grid block x, y, yaw index
        self.traj = traj  # trajectory x, y
        self.curvature = curvature  # steering angle throughout the trajectory
        self.cost = cost  # node cost
        self.parent_index = parent_index  # parent node index
        self.direction = direction

    def get_hybrid_index(self):
        return tuple([self.grid_index[0], self.grid_index[1], self.grid_index[2]])


class HybridAStarSearch(object):
    # cost of the hybrid search
    STEER_COST = 1
    DELTA_STEER_COST = 5
    DEVIATION_COST = 1
    DISTANCE_COST = 1
    DIRECTION_CHANGE_COST = 1000
    REVERSE_COST = 5000
    HYBRID_COST = 50
    # search parameters
    MIN_LENGTH_TO_GOAL = 1000

    def __init__(
        self,
        start_pose: list,
        goal_pose: list,
        config_environment,
        car_model: CarModel,
        search_heuristic,
        motion_type="Pawn",
        yaw_resolution=math.radians(10),
        plan_resolution=0.1,
    ):
        """Class for hybrid a star search
        Args:
            start_pose: [x, y, heading] in baselink
            end_pose: [x, y, heading] in baselink
            config_env: instance of configuration space for searching
            car_model: collision and kinematic model of vehicle
            heuristic_function: search heuristic of hybrid a star
            plan_resolution: resolution of planned map
            yaw_resolution: resolution of planned yaws
        Returns:
            None
        """
        self.plan_resolution = plan_resolution
        self.yaw_resolution = yaw_resolution
        # configuration space environment of searching
        self.config_env = config_environment
        # kinematic and collision model of the car
        self.car_model = car_model
        # search heuristic
        self.search_heuristic = search_heuristic
        self.motion_type = motion_type
        self.initialize_expand_utils()
        self.motion_steers = self.get_motion_steers()
        self.init_time_statistics()
        self.start_node = self.init_node(start_pose)
        self.goal_node = self.init_node(goal_pose)

    def check_start_goal_node_feasibility(self):
        start_is_not_ok = self.check_collision(self.start_node.traj)
        goal_is_not_ok = self.check_collision(self.goal_node.traj)

        return start_is_not_ok or goal_is_not_ok

    def calculate_node_index(self, x, y, yaw):
        hybrid_index = (
            round(x / self.plan_resolution),
            round(y / self.plan_resolution),
            round(yaw / self.yaw_resolution),
        )

        return hybrid_index

    def init_time_statistics(self):
        self.collision_check_duration = 0
        self.kinematic_expansion_duration = 0
        self.node_cost_duration = 0

    def initialize_expand_utils(self):
        if self.motion_type == "Pawn":
            # motion steers
            self.get_motion_steers = self._get_motion_steers_dubins
            # goal extension
            self.get_goal_extension_path = self._get_goal_extension_with_dubins_path
        if self.motion_type == "King":
            # motion steers
            self.get_motion_steers = self._get_motion_steers_reeds_shepp
            # goal extension
            self.get_goal_extension_path = (
                self._get_goal_extension_with_reeds_shepp_path
            )

    def init_node(self, pose: list) -> Node:
        """get hybrid a star search node
        Args:
            pose: 2d pose of the robot
        Returns:
            pose_node: node of the robot
        """
        x, y, yaw = (
            pose[0],
            pose[1],
            pose[2],
        )
        hybrid_index = self.calculate_node_index(x, y, yaw)
        # actual pose of the robot
        pose = [x, y, yaw]
        pose_node = Node(hybrid_index, [pose], [0], 0, [1], hybrid_index)

        return pose_node

    def calculate_reeds_shepp_path_cost(self, current_node: Node, path: rs_curves.PATH):
        # Previos Node Cost
        cost = current_node.cost

        # Distance cost
        path_lengths = np.array(path.lengths)
        idxs = np.where(path_lengths < 0)[0]
        other_move_cost = len(path_lengths) - len(idxs)
        cost += self.REVERSE_COST * len(idxs) + other_move_cost

        # direction change
        length_before = np.array(path_lengths[:-1])
        length_after = np.array(path_lengths[1:])
        direction_changes = length_before * length_after
        idxs = np.where(direction_changes < 0)
        cost += len(idxs) * self.DIRECTION_CHANGE_COST

        # Steering Angle Cost
        path_types = np.array(path.ctypes)
        idxs = np.where(path_types != "S")
        cost += self.car_model.MAX_STEER * self.STEER_COST * len(idxs)

        # Steering Angle change cost
        steers = np.zeros(len(path_types))
        right_steer_idxs = np.where(path_types == "R")[0]
        left_steer_idxs = np.where(path_types == "WB")[0]
        steers[right_steer_idxs] = -self.car_model.MAX_STEER
        steers[left_steer_idxs] = self.car_model.MAX_STEER
        delta_steers = np.abs(np.diff(steers))
        cost += np.sum(delta_steers)

        return cost

    def calculate_dubins_path_cost(self, current_node, path: np.ndarray):
        """get hybrid a star search node
        Args:
            current node: hybrid a star node
            path: numpy array of path with element of [x,y,yaw]
        Returns:
            cost: cost of appending this path
            path_length: length of the path
        """
        # Previos Node Cost
        cost = current_node.cost

        # distance cost
        path_length = calculate_path_length(path[:, 0], path[:, 1])
        cost += path_length * self.DISTANCE_COST

        # steer cost
        delta_yaw = angle_wrap(np.max(path[:, -1]) - np.min(path[:, -1]))
        cost += delta_yaw * self.STEER_COST

        return cost, path_length

    def _get_goal_extension_with_dubins_path(self, current_node: Node):
        """get hybrid a star search node
        Args:
            current node: hybrid a star node
        Returns:
            goal_node
        """
        # Get x, y, yaw of current_node and goal_node
        start_x, start_y, start_yaw = (
            current_node.traj[-1][0],
            current_node.traj[-1][1],
            current_node.traj[-1][2],
        )

        goal_x, goal_y, goal_yaw = (
            self.goal_node.traj[-1][0],
            self.goal_node.traj[-1][1],
            self.goal_node.traj[-1][2],
        )

        # Instantaneous Curvature
        dubin_path = self.get_dubins_path(
            start_x,
            start_y,
            start_yaw,
            goal_x,
            goal_y,
            goal_yaw,
            self.car_model.curvature,
        )

        cost = self.calculate_dubins_path_cost(current_node, dubin_path)
        traj = np.copy(dubin_path[:, :3])
        traj[:, -1] = angle_wrap(traj[:, -1])
        path_length = calculate_path_length(dubin_path[:, 0], dubin_path[:, 1])
        ks = list(dubin_path[:, 3])
        if not self.check_collision(traj) and path_length < self.MIN_LENGTH_TO_GOAL:
            return Node(
                self.goal_node.get_hybrid_index(),
                traj,
                ks,
                cost,
                np.ones_like(ks).tolist(),
                current_node.get_hybrid_index(),
            )

        return None

    def _get_goal_extension_with_reeds_shepp_path(self, current_node: Node):
        # Get x, y, yaw of currentNode and goalNode
        start_x, start_y, start_yaw = (
            current_node.traj[-1][0],
            current_node.traj[-1][1],
            current_node.traj[-1][2],
        )

        goal_x, goal_y, goal_yaw = (
            self.goal_node.traj[-1][0],
            self.goal_node.traj[-1][1],
            self.goal_node.traj[-1][2],
        )

        # Instantaneous Curvature

        #  Find all possible reeds-shepp paths if self.motion_type == "King":
        to_goal_paths = rs_curves.calc_all_paths(
            start_x,
            start_y,
            start_yaw,
            goal_x,
            goal_y,
            goal_yaw,
            self.car_model.curvature,
            self.plan_resolution,
        )

        # Check if reedsSheppPaths is empty
        if not to_goal_paths:
            return None

        # Find path with lowest cost considering non-holonomic constraints
        cost_queue = heapdict()
        for path in to_goal_paths:
            cost_queue[path] = self.calculate_reeds_shepp_path_cost(current_node, path)

        # Find first path in priority queue that is collision free
        while len(cost_queue) != 0:
            path, path_cost = cost_queue.popitem()
            # print("path length: ", path.L)
            traj = np.array([path.x, path.y, path.yaw]).T
            # steers = list(np.arctan((np.array(path.cs) * self.car_model.WHEEL_BASE)))
            # print(len(traj), len(path.directions), len(path.cs))
            if not self.check_collision(traj) and path.L < self.MIN_LENGTH_TO_GOAL:
                cost = path_cost
                return Node(
                    self.goal_node.get_hybrid_index(),
                    traj,
                    path.cs,
                    cost,
                    path.directions,
                    current_node.get_hybrid_index(),
                )

        return None

    def get_dubins_path(
        self, start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, curvature
    ):
        start_pose = [start_x, start_y, start_yaw]
        goal_pose = [goal_x, goal_y, goal_yaw]
        path = dubins.shortest_path(start_pose, goal_pose, 1.0 / curvature)
        dubins_path, _ = path.sample_many(self.plan_resolution)
        dubins_path = np.vstack(
            [np.array(dubins_path), np.array([[goal_x, goal_y, goal_yaw]])]
        )
        xs, ys, yaws, ks, _ = calc_spline_course(
            dubins_path[:, 0], dubins_path[:, 1], ds=self.plan_resolution
        )
        dubins_path = np.array([xs, ys, yaws, ks]).T

        return dubins_path

    def simulated_path_cost(self, current_node: Node, traj, motion_command):
        # Previos Node Cost
        cost = current_node.cost

        path_length = calculate_path_length(traj[:, 0], traj[:, 1])
        # Distance cost
        cost += path_length

        # reverse cost
        if motion_command[1] == -1:
            cost += self.REVERSE_COST

        # Steering Angle Cost
        cost += motion_command[0] * self.STEER_COST

        # Steering Angle change cost
        steer_angle = math.atan(current_node.curvature[0] * self.car_model.WHEEL_BASE)
        cost += abs(motion_command[0] - steer_angle) * self.DELTA_STEER_COST

        # direction change cost
        if current_node.direction[0] != motion_command[1]:
            cost += self.DIRECTION_CHANGE_COST

        return cost

    def _get_motion_steers_dubins(self):
        steer_ranges = np.arange(
            self.car_model.MAX_STEER,
            -(self.car_model.MAX_STEER + self.yaw_resolution),
            -self.yaw_resolution,
        )

        directions = np.ones_like(steer_ranges)
        motion_steers = np.vstack((steer_ranges, directions)).T

        return motion_steers

    def _get_motion_steers_reeds_shepp(self):
        steer_ranges = np.arange(
            self.car_model.MAX_STEER,
            -(self.car_model.MAX_STEER + self.yaw_resolution / 2.0),
            -self.yaw_resolution / 2.0,
        )

        directions = np.ones_like(steer_ranges)
        directions[1 : len(directions) : 2] = -1
        motion_steers = np.vstack((steer_ranges, directions)).T

        return motion_steers

    # TODO: build up motion libraries for table looking up
    def kinematic_simulation_node(self, current_node: Node, moition_command: list):
        """get hybrid a star search node
        Args:
            current_node: hybrid a star node
            motion_command: [steer, direction]
        Returns:
            goal_node
        """
        init = time.time()
        steer_angle = moition_command[0]
        speed_direction = moition_command[1]
        search_length = self.search_heuristic.get_search_length(current_node.traj[-1])
        num_steps = round(search_length / self.plan_resolution)
        yaw_step = (
            speed_direction
            * self.plan_resolution
            / self.car_model.WHEEL_BASE
            * math.tan(steer_angle)
        )

        init_yaw = angle_wrap(current_node.traj[-1][2] + yaw_step)
        init_x = current_node.traj[-1][0]
        init_y = current_node.traj[-1][1]

        yaws = np.linspace(
            init_yaw, init_yaw + yaw_step * (num_steps + 1), num_steps + 2
        )
        yaws = angle_wrap(yaws)
        xs = self.plan_resolution * np.cos(yaws[:-1]) * speed_direction
        xs = init_x + np.cumsum(xs)
        ys = self.plan_resolution * np.sin(yaws[:-1]) * speed_direction
        ys = init_y + np.cumsum(ys)

        traj = np.vstack([xs, ys, yaws[1:]]).T
        duration = time.time() - init
        self.kinematic_expansion_duration += duration

        grid_index = self.calculate_node_index(traj[-1][0], traj[-1][1], traj[-1][2])

        # Check if node is valid
        if self.check_collision(traj):
            return None

        # Calculate Cost of the node
        cost = self.simulated_path_cost(current_node, traj, moition_command)
        curvature = np.tan(moition_command[0]) / self.car_model.WHEEL_BASE
        return Node(
            grid_index,
            traj,
            [curvature] * len(traj),
            cost,
            [moition_command[1]] * len(traj),
            current_node.get_hybrid_index(),
        )

    def check_collision(self, path):
        init = time.time()
        collision = False
        # check if the path collides with the geo environment
        path_is_feasible = self.config_env.check_path_feasibility(self.car_model, path)

        # check if the path in the search area
        path_is_in_search_range = self.search_heuristic.check_path_feasibility(
            self.car_model, path
        )

        collision = (not path_is_feasible) or (not path_is_in_search_range)

        self.collision_check_duration += time.time() - init

        return collision

    def get_path_from_expanded_nodes(self, closed_set: dict):
        # Goal Node data
        start_node_index = self.start_node.get_hybrid_index()
        current_node_index = self.goal_node.parent_index
        if not current_node_index in closed_set.keys():
            return [], [], [], [], []
        current_node = closed_set[current_node_index]
        xs = []
        ys = []
        yaws = []
        dirs = []
        ks = []

        # Iterate till we reach start node from goal node
        while current_node_index != start_node_index:
            a, b, c = zip(*current_node.traj)
            xs += a[::-1]
            ys += b[::-1]
            yaws += c[::-1]
            dirs += current_node.direction[::-1]
            ks += current_node.curvature[::-1]
            # print("dirs, traj: ", len(current_node.direction), len(current_node.traj))
            current_node_index = current_node.parent_index
            current_node = closed_set[current_node_index]

        return xs[::-1], ys[::-1], yaws[::-1], ks[::-1], dirs[::-1]

    def get_heuristic_cost(self, pose):
        init = time.time()
        # calculate the cost from heuristic function
        cost = self.search_heuristic.calculate_state_cost(pose)
        self.node_cost_duration += time.time() - init

        return cost

    def check_the_arrival(self, goal_extension_node, current_node: Node):
        goal_node = None
        if goal_extension_node is not None:
            # path found
            goal_node = goal_extension_node

        # check if it is very near to the goal pose
        dist_to_goal = np.hypot(
            current_node.traj[-1][1] - self.goal_node.traj[0][1],
            current_node.traj[-1][0] - self.goal_node.traj[0][0],
        )

        x_dist = np.abs(current_node.traj[-1][0] - self.goal_node.traj[0][0])
        y_dist = np.abs(current_node.traj[-1][1] - self.goal_node.traj[0][1])
        yaw_diff = np.abs(
            angle_wrap(current_node.traj[-1][2] - self.goal_node.traj[0][2])
        )

        # if current_node.get_hybrid_index() == self.goal_node.get_hybrid_index():
        if (
            x_dist < self.plan_resolution
            and y_dist < self.plan_resolution
            and yaw_diff < self.yaw_resolution
        ):
            print(
                "The goal position has some difference to the desired one, to goal distance %.2f and yaw difference is %.2f "
                % (dist_to_goal, math.degrees(yaw_diff))
            )
            goal_node = current_node
            goal_node.grid_index = self.goal_node.grid_index

        return goal_node

    def hybrid_a_star_search(self, plt=None, max_nodes=2000):
        init = time.time()
        # Add start node to open Set
        open_set = {self.start_node.get_hybrid_index(): self.start_node}
        closed_set = {}

        # Create a priority queue for acquiring nodes based on their cost's
        cost_queue = heapdict()

        # Add start mode into priority queue
        cost_queue[self.start_node.get_hybrid_index()] = max(
            self.start_node.cost,
            self.HYBRID_COST * self.get_heuristic_cost(self.start_node.traj[-1]),
        )
        counter = 0
        init = time.time()
        trajs = []

        # check start and goal before searching
        start_or_goal_is_not_ok = self.check_start_goal_node_feasibility()
        if start_or_goal_is_not_ok:
            print("start or goal position is interfere with obstacles!!")
            return [], [], [], [], [], 0

        if self.start_node.get_hybrid_index() == self.goal_node.get_hybrid_index():
            print("start and goal position is the same!!")

        # Run loop until path is found or open set is empty
        while True:
            if counter > max_nodes:
                # TODO: output a feasible path nearest to the goal point
                print("drop the planner")
                break
            counter += 1
            if counter % 200 == 0:
                print(
                    "%d of nodes expanded, %f(s) time consumed"
                    % (counter, time.time() - init)
                )
            # Check if openSet is empty, if empty no solution available
            if not open_set:
                print("No solution is available")
                break

            # Get first node in the priority queue
            current_node_idx, _ = cost_queue.popitem()
            current_node = open_set[current_node_idx]
            open_set.pop(current_node_idx)
            closed_set[current_node_idx] = current_node

            # exit as long as a kinematically feasible path is found
            goal_extended_node = self.get_goal_extension_path(current_node)

            goal_node = self.check_the_arrival(goal_extended_node, current_node)

            # the path is found, end the searching
            if goal_node is not None:
                closed_set[goal_node.get_hybrid_index()] = goal_node
                break

            # Get all simulated Nodes from current node
            for i in range(len(self.motion_steers)):
                simulated_node = self.kinematic_simulation_node(
                    current_node, self.motion_steers[i]
                )

                # Check if path is within map bounds and is collision free
                if not simulated_node:
                    continue

                # for debugging
                trajs.append(simulated_node.traj)
                # visualization of path
                if plt is not None:
                    x, y, z = zip(*simulated_node.traj)
                    if simulated_node.direction[0] == 1:
                        plt.plot(x, y, linewidth=0.3, color="g")
                    if simulated_node.direction[0] == -1:
                        plt.plot(x, y, linewidth=0.3, color="r")
                    # self.car_model.draw_car(plt, x[-1], y[-1], z[-1])

                # Check if simulated node is already in closed set
                simulated_node_index = simulated_node.get_hybrid_index()
                if simulated_node_index not in closed_set:
                    # Check if simulated node is already in open set, if not add it open set as well as in priority queue
                    if simulated_node_index not in open_set:
                        open_set[simulated_node_index] = simulated_node
                        cost_queue[simulated_node_index] = max(
                            simulated_node.cost,
                            self.HYBRID_COST
                            * self.get_heuristic_cost(simulated_node.traj[-1]),
                        )
                    else:
                        if simulated_node.cost < open_set[simulated_node_index].cost:
                            open_set[simulated_node_index] = simulated_node
                            cost_queue[simulated_node_index] = max(
                                simulated_node.cost,
                                self.HYBRID_COST
                                * self.get_heuristic_cost(simulated_node.traj[-1]),
                            )

        # Backtrack
        hybrid_search_time = time.time() - init
        # print("kinematic_expansion time: ", self.kinematic_expansion_duration)
        # print("collision check time: ", self.collision_check_duration)
        # print("node cost time: ", self.node_cost_duration)
        print("hybrid search time: ", hybrid_search_time)
        print("counter of nodes: ", counter)
        x, y, yaw, ks, dirs = self.get_path_from_expanded_nodes(closed_set)
        # print(len(x), len(y), len(yaw), len(ks), len(dirs))
        return (x, y, yaw, dirs, ks, counter)
