# Convert ROS Occupancy Grid to grid map
from tokenize import Double
from turtle import position
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
import scipy.spatial.kdtree as kd
from transformations import euler_from_quaternion


class GridMapFeatures:
    def __init__(self):
        self.obstacles_boolean = np.array([])  # grid fillings for the obstalces
        self.obstacles_xs = np.array([])  # positions of obstalces in baselink
        self.obstacles_ys = np.array([])  # positions of obstacles in baselink
        self.obstacles_kd_tree = None  # kd tree for obstacles
        self.obstacle_field_map = None
        self.odom_trans_map = None


def convert_ros_occupancy_grid_to_ndarray(occupancy_grid_msg: OccupancyGrid):

    resolution = occupancy_grid_msg.info.resolution
    width = occupancy_grid_msg.info.width
    height = occupancy_grid_msg.info.height
    # the orientation is 0
    position_x = -occupancy_grid_msg.info.origin.position.x
    position_y = -occupancy_grid_msg.info.origin.position.y
    # quaternion = [
    #     occupancy_grid_msg.info.origin.orientation.w,
    #     occupancy_grid_msg.info.origin.orientation.x,
    #     occupancy_grid_msg.info.origin.orientation.y,
    #     occupancy_grid_msg.info.origin.orientation.z,
    # ]
    # yaw_angle = euler_from_quaternion(quaternion)[-1]
    map_2d_position = (position_x, position_y)
    occupancy_map_array = np.array(occupancy_grid_msg.data).reshape((width, height)).T

    return occupancy_map_array, resolution, map_2d_position
    # return occupancy_map_array, resolution


def shrink_grid_map(
    occupancy_map: np.ndarray, old_resolution: float, new_resolution: float
):

    if new_resolution < old_resolution:

        return occupancy_map

    width, height = occupancy_map.shape
    resized_width = int(width * (old_resolution / new_resolution))
    resized_height = int(height * (old_resolution / new_resolution))

    shrinked_map = cv2.resize(
        occupancy_map,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST_EXACT,
    )

    return shrinked_map


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get("padder", 10)
    vector[: pad_width[0]] = pad_value
    vector[-pad_width[1] :] = pad_value


def get_grid_map_features(
    occupancy_map: np.ndarray, resolution: float, map_2d_position: tuple
):
    origin_x, origin_y = map_2d_position
    obstacle_value = np.max(occupancy_map)

    # padded occupancy map with boundary
    occupancy_map_array_padded = np.pad(occupancy_map, 1, pad_with, padder=0)

    # obstacle idex
    obstacle_idxs = np.where(occupancy_map_array_padded == obstacle_value)
    # obstacle positions
    # obstacles_xs = obstacle_idxs[0] * resolution - origin_x
    # obstacles_ys = obstacle_idxs[1] * resolution - origin_y
    obstacles_xs = obstacle_idxs[0] * resolution + resolution / 2.0
    obstacles_ys = obstacle_idxs[1] * resolution + resolution / 2.0
    # generate kd tree of obstacles
    obstacle_kd_tree = kd.KDTree(np.array([obstacles_xs, obstacles_ys]).T)

    # boolean map
    obstacles = np.copy(occupancy_map_array_padded).astype("bool")
    obstacles[:, :] = False
    obstacles[obstacle_idxs] = True

    grid_map_features = GridMapFeatures()
    grid_map_features.obstacles_boolean = obstacles
    grid_map_features.obstacles_kd_tree = obstacle_kd_tree
    grid_map_features.obstacles_xs = obstacles_xs
    grid_map_features.obstacles_ys = obstacles_ys
    grid_map_features.obstacle_field_map = occupancy_map_array_padded

    return grid_map_features
