from car_model import CarModel
import numpy as np
import math
from utils.navigation_utils import angle_wrap, convert_2d_xys_to_target_frame
import utils.transformation as trans
from safety_forward_path_plan import (
    get_dubins_path_full,
)


def get_y_type_parking_path_in_odom(
    car: CarModel,
    start_pose,
    end_pose,
    backward_distance=5.0,
    forward_distance=2.5,
    backward_steer=0.1,
    forward_steer=0.5,
    step_size=0.1,
):
    backward_steer_dir = get_backward_steer_dir_for_y_type_parking(start_pose, end_pose)
    forward_steer_dir = -backward_steer_dir
    y_type_path_in_baselink = get_y_type_parking_path(
        car,
        backward_distance,
        backward_steer * backward_steer_dir,
        forward_distance,
        forward_steer * forward_steer_dir,
        step_size,
    )
    odom_T_baselink = trans.states2SE3([end_pose[0], end_pose[1], 0, 0, 0, end_pose[2]])
    y_type_path_in_odom = get_path_in_odom(odom_T_baselink, y_type_path_in_baselink)

    return y_type_path_in_odom


def get_backward_steer_dir_for_y_type_parking(start_pose, end_pose):
    # right turn
    if end_pose[1] - start_pose[1] > 0:
        backward_steer = np.sign(1 * math.cos(start_pose[2]))
    # left turn
    else:
        backward_steer = np.sign(-1 * math.cos(start_pose[2]))

    return backward_steer


# calculate the backward path
def calculate_motion_path(init_pose, motion_command, search_length, wheel_base, step):
    steer_angle = motion_command[0]
    speed_direction = motion_command[1]
    num_steps = round(search_length / step)
    yaw_step = speed_direction * step / wheel_base * math.tan(steer_angle)

    init_yaw = angle_wrap(init_pose[-1] + yaw_step)
    init_x = init_pose[0]
    init_y = init_pose[1]

    yaws = np.linspace(init_yaw, init_yaw + yaw_step * (num_steps), num_steps + 1)
    yaws = angle_wrap(yaws)
    xs = step * np.cos(yaws[:-1]) * speed_direction
    xs = init_x + np.cumsum(xs)
    ys = step * np.sin(yaws[:-1]) * speed_direction
    ys = init_y + np.cumsum(ys)

    path = np.vstack([xs, ys, yaws[1:]]).T
    path = np.vstack([init_pose, path])

    # add curvature and dirs
    curvature = 0
    if abs(motion_command[0]) > 0.00001:
        curvature = math.tan(motion_command[0]) / wheel_base

    ks = np.ones((len(path), 1)) * curvature
    dirs = np.ones((len(path), 1)) * motion_command[1]
    path = np.hstack((path, ks, dirs))

    return path


def get_y_type_parking_path(
    car_model: CarModel,
    backward_length,
    backward_steer,
    forward_length,
    forward_steer,
    step,
):
    back_path = calculate_motion_path(
        [0, 0, 0],
        [backward_steer, -1],
        backward_length,
        car_model.WHEEL_BASE,
        step,
    )

    forward_path = calculate_motion_path(
        back_path[-1, :3],
        [forward_steer, 1],
        forward_length,
        car_model.WHEEL_BASE,
        step,
    )
    back_path[:, -1] = 1
    back_path = back_path[::-1]
    forward_path[:, -1] = -1
    forward_path = forward_path[::-1]
    path = np.vstack([forward_path, back_path])

    return path


def get_path_in_odom(odom_T_baselink, path, plt=None):
    current_pose = trans.SE32states(odom_T_baselink)
    path_in_odom = np.copy(path)
    path_in_odom[:, 2] += current_pose[-1]
    path_in_odom[:, 0], path_in_odom[:, 1] = convert_2d_xys_to_target_frame(
        path_in_odom[:, 0], path_in_odom[:, 1], odom_T_baselink
    )

    return path_in_odom


def get_warm_start_path_y_type(
    car: CarModel,
    start_pose,
    end_pose,
    steer_backward,
    forward_distance,
    backward_distance,
    steer_forward,
    step_size,
):
    y_type_path_in_odom = get_y_type_parking_path_in_odom(
        car,
        start_pose,
        end_pose,
        backward_distance=backward_distance,
        forward_distance=forward_distance,
        backward_steer=steer_backward,
        forward_steer=steer_forward,
        step_size=step_size,
    )

    # forward path
    intermediate_pose = y_type_path_in_odom[0][:3]

    # dubins path
    to_intermediate_pose_path = get_dubins_path_full(
        start_pose,
        intermediate_pose,
        turning_radius=1.0 / car.curvature,
        step_size=step_size,
    )

    path = np.vstack([to_intermediate_pose_path, y_type_path_in_odom])
    path_xs = path[:, 0]
    path_ys = path[:, 1]
    path_yaws = path[:, 2]
    path_ks = path[:, 3]
    dirs = path[:, 4]

    return path_xs, path_ys, path_yaws, path_ks, dirs


def get_warm_start_path_dubins(car: CarModel, start_pose, end_pose, step_size):
    path = get_dubins_path_full(
        start_pose,
        end_pose,
        turning_radius=1.0 / car.curvature,
        step_size=step_size,
    )

    return path[:, 0], path[:, 1], path[:, 2], path[:, 3], path[:, 4]
