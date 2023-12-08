from utils.cubic_spline import calc_spline_course
from orchard_geometry_environment import OrchardGeometryEnvironment
from car_model import CarModel
from reference_line_heuristic import ReferenceLineHeuristic
import copy
import numpy as np
import math
from utils.navigation_utils import angle_wrap, convert_2d_xys_to_target_frame
import utils.transformation as trans
from hybrid_a_star_search import HybridAStarSearch
from safety_forward_path_plan import (
    get_offset_poses_for_row_traversing,
    get_dubins_path_full,
    get_base_pose,
    LEAVE_POSE,
    FAR_SIDE,
    NEAR_SIDE,
)

ERROR_CODE_FOR_NONE = -1
ERROR_CODE_FOR_Y_PARKING = 0
ERROR_CODE_FOR_HYBRID_A_STAR = 1


def get_side_for_a_pose(current_pose, map_tree_rows):
    row_centers = np.mean(map_tree_rows[:, :, :], axis=1)
    coeff = np.polyfit(row_centers[:, 0], row_centers[:, 1], deg=1)
    map_center_line_k, map_center_line_b = coeff[0], coeff[1]
    pose_x = current_pose[0]
    pose_y = current_pose[1]
    position_sign = np.sign(map_center_line_k * pose_x + map_center_line_b - pose_y)
    origin_sign = np.sign(0 * map_center_line_k + map_center_line_b - 0)

    side_idx = NEAR_SIDE if position_sign == origin_sign else FAR_SIDE

    return side_idx


def get_current_row_id(current_pose, map_tree_rows):
    side = get_side_for_a_pose(current_pose, map_tree_rows)
    if side == NEAR_SIDE:
        side_idx = 0
    else:
        side_idx = 1
    row_num = len(map_tree_rows)
    drive_row_ys = map_tree_rows[: row_num - 1, side_idx, 1] + np.diff(
        map_tree_rows[:, side_idx, 1]
    )
    row_id = np.argmin(np.abs((current_pose[1] - drive_row_ys)))

    return row_id, side


# planner to combine forward and y park
def headland_planner_y_type_park_combined(
    config_env: OrchardGeometryEnvironment,
    car_model: CarModel,
    start_pose: list,
    end_pose: list,
    motion_type="Pawn",
    drive_row_offset=4.5,
    max_steer_backward=0.35,
    min_steer_backward=0.22,
    max_steer_forward=0.55,
    min_steer_forward=0.50,
    max_backward_distance=2.0,
    min_forward_distance=1.4,
    max_forward_distance=2.5,
    min_backward_distance=0.7,
    step_size=0.1,
    tree_width_in_forward_plan=0.2,
    max_steer_for_offset_plan=0.5,
):
    # do the forward planning
    ## this is for the safety
    config_env_plan = copy.deepcopy(config_env)
    config_env_plan.update_tree_width(tree_width_in_forward_plan)
    start_exit_pose, end_enter_pose = get_offset_poses_for_row_traversing(
        start_pose,
        end_pose,
        car_model,
        config_env_plan,
        max_steer_angle=max_steer_for_offset_plan,
    )
    turn_radius = car_model.get_turn_radius(max_steer_angle=None)
    path = get_dubins_path_full(
        start_exit_pose, end_enter_pose, turn_radius, step_size=step_size
    )
    # check if the path is feasible
    if config_env_plan.check_path_feasibility(car_model, path, boundary_check=True):
        print("got a path forward driving")
        path_xs, path_ys, path_yaws, path_ks, dirs = (
            path[:, 0],
            path[:, 1],
            path[:, 2],
            path[:, 3],
            path[:, 4],
        )
        return ERROR_CODE_FOR_NONE, path_xs, path_ys, path_yaws, path_ks, dirs

    # do the y type
    error_code, path_xs, path_ys, path_yaws, path_ks, dirs = headland_planner_y_type_park(
        config_env=config_env,
        car_model=car_model,
        start_pose=start_pose,
        end_pose=end_pose,
        motion_type=motion_type,
        drive_row_offset=drive_row_offset,
        max_steer_backward=max_steer_backward,
        min_steer_backward=min_steer_backward,
        max_steer_forward=max_steer_forward,
        min_steer_forward=min_steer_forward,
        max_backward_distance=max_backward_distance,
        min_forward_distance=min_forward_distance,
        max_forward_distance=max_forward_distance,
        min_backward_distance=min_backward_distance,
        step_size=step_size,
        debug=False,
    )

    return error_code, path_xs, path_ys, path_yaws, path_ks, dirs


def headland_planner_y_type_park(
    config_env: OrchardGeometryEnvironment,
    car_model: CarModel,
    start_pose: list,
    end_pose: list,
    motion_type="Pawn",
    drive_row_offset=4.5,
    max_steer_backward=0.35,
    min_steer_backward=0.22,
    max_steer_forward=0.55,
    min_steer_forward=0.50,
    max_backward_distance=2.0,
    min_forward_distance=1.4,
    max_forward_distance=2.5,
    min_backward_distance=0.7,
    step_size=0.1,
    debug=True,
):
    error_code = ERROR_CODE_FOR_NONE
    parameters = []
    # Y type parking
    backward_steer_dir = get_backward_steer_dir_for_y_type_parking(start_pose, end_pose)
    forward_steer_dir = -backward_steer_dir
    # create y-type parking path
    print("start Y park searching")
    if debug:
        y_type_parking_path, parameters = search_y_type_parking_path(
            car_model,
            config_env,
            end_pose,
            backward_steer_dir,
            forward_steer_dir,
            max_steer_backward,
            max_steer_forward,
            max_backward_distance,
            max_forward_distance,
            min_forward_distance,
            min_backward_distance,
            min_steer_backward,
            min_steer_forward,
            step_size=step_size,
            debug=debug,
        )
    else:
        y_type_parking_path = search_y_type_parking_path(
            car_model,
            config_env,
            end_pose,
            backward_steer_dir,
            forward_steer_dir,
            max_steer_backward,
            max_steer_forward,
            max_backward_distance,
            max_forward_distance,
            min_forward_distance,
            min_backward_distance,
            min_steer_backward,
            min_steer_forward,
            step_size=step_size,
            debug=debug,
        )

    if len(y_type_parking_path) == 0:
        error_code = ERROR_CODE_FOR_Y_PARKING
        print("No enough space on headland for the tractor to enter the row!!")
        if debug:
            return error_code, parameters, 0, [], [], [], [], []
        else:
            return error_code, [], [], [], [], []

    # forward path
    intermediate_pose = y_type_parking_path[0][:3]
    # get topological waypoints for headland traveling
    way_points = config_env.get_topology_waypoints(
        start_pose, intermediate_pose, drive_row_offset=drive_row_offset
    )

    search_heuristic = ReferenceLineHeuristic(way_points, intermediate_pose, car_model)

    planner_to_intermediate_pose = HybridAStarSearch(
        start_pose,
        intermediate_pose,
        config_env,
        car_model,
        search_heuristic,
        motion_type=motion_type,
        plan_resolution=step_size,
    )
    (
        drive_to_enter_row_xs,
        drive_to_enter_row_ys,
        drive_to_enter_row_yaws,
        drive_to_enter_row_dirs,
        drive_to_enter_row_ks,
        a_star_node_couts,
    ) = planner_to_intermediate_pose.hybrid_a_star_search(max_nodes=400)

    if len(drive_to_enter_row_xs) == 0:
        print("cannot search a path to parking start pose!!")
        error_code = ERROR_CODE_FOR_HYBRID_A_STAR
        if debug:
            return error_code, parameters, a_star_node_couts, [], [], [], [], []
        else:
            return error_code, [], [], [], [], []

    # (
    #     drive_to_enter_row_xs,
    #     drive_to_enter_row_ys,
    #     drive_to_enter_row_yaws,
    #     drive_to_enter_row_ks,
    #     _,
    # ) = calc_spline_course(drive_to_enter_row_xs, drive_to_enter_row_ys, ds=step_size)
    # drive_to_enter_row_dirs = np.ones_like(drive_to_enter_row_xs)
    path_xs = np.concatenate([drive_to_enter_row_xs, y_type_parking_path[:, 0]])
    path_ys = np.concatenate([drive_to_enter_row_ys, y_type_parking_path[:, 1]])
    path_yaws = np.concatenate([drive_to_enter_row_yaws, y_type_parking_path[:, 2]])
    path_ks = np.concatenate([drive_to_enter_row_ks, y_type_parking_path[:, 3]])
    dirs = np.concatenate([drive_to_enter_row_dirs, y_type_parking_path[:, 4]])

    if debug:
        return (
            error_code,
            parameters,
            a_star_node_couts,
            path_xs,
            path_ys,
            path_yaws,
            path_ks,
            dirs,
        )
    else:
        return error_code, path_xs, path_ys, path_yaws, path_ks, dirs


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


def get_row_enter_path(
    config_env: OrchardGeometryEnvironment,
    car_model: CarModel,
    start_pose: list,
    end_pose: list,
    drive_row_offset=4.5,
    steer_backward=0.35,
    steer_forward=0.55,
    backward_distance=2.0,
    forward_distance=2.5,
    step_size=0.1,
):
    # Y type parking
    error_code = ERROR_CODE_FOR_NONE
    backward_steer_dir = get_backward_steer_dir_for_y_type_parking(start_pose, end_pose)
    forward_steer_dir = -backward_steer_dir
    human_type_path_in_baselink = get_y_type_parking_path(
        car_model,
        backward_distance,
        steer_backward * backward_steer_dir,
        forward_distance,
        steer_forward * forward_steer_dir,
        step_size,
    )
    odom_T_baselink = trans.states2SE3([end_pose[0], end_pose[1], 0, 0, 0, end_pose[2]])
    y_type_parking_path = get_path_in_odom(odom_T_baselink, human_type_path_in_baselink)
    if not config_env.check_path_feasibility(car_model, y_type_parking_path):
        print("not enough space for row entering")
        error_code = ERROR_CODE_FOR_Y_PARKING
    # forward path
    intermediate_pose = y_type_parking_path[0][:3]

    # get topological waypoints for headland traveling
    way_points = config_env.get_topology_waypoints(
        start_pose, intermediate_pose, drive_row_offset=drive_row_offset
    )

    search_heuristic = ReferenceLineHeuristic(way_points, intermediate_pose, car_model)

    planner_to_intermediate_pose = HybridAStarSearch(
        start_pose,
        intermediate_pose,
        config_env,
        car_model,
        search_heuristic,
        motion_type="Pawn",
        plan_resolution=step_size,
    )
    (
        drive_to_enter_row_xs,
        drive_to_enter_row_ys,
        drive_to_enter_row_yaws,
        drive_to_enter_row_dirs,
        _,
        a_star_node_couts,
    ) = planner_to_intermediate_pose.hybrid_a_star_search(max_nodes=400)

    if len(drive_to_enter_row_xs) == 0:
        print("cannot search a path to parking start pose!!")
        return (
            ERROR_CODE_FOR_HYBRID_A_STAR,
            a_star_node_couts,
            y_type_parking_path[:, 0],
            y_type_parking_path[:, 1],
            y_type_parking_path[:, 2],
            y_type_parking_path[:, 3],
            y_type_parking_path[:, 4],
        )

    (
        drive_to_enter_row_xs,
        drive_to_enter_row_ys,
        drive_to_enter_row_yaws,
        drive_to_enter_row_ks,
        _,
    ) = calc_spline_course(drive_to_enter_row_xs, drive_to_enter_row_ys, ds=step_size)
    drive_to_enter_row_dirs = np.ones_like(drive_to_enter_row_xs)
    path_xs = np.concatenate([drive_to_enter_row_xs, y_type_parking_path[:, 0]])
    path_ys = np.concatenate([drive_to_enter_row_ys, y_type_parking_path[:, 1]])
    path_yaws = np.concatenate([drive_to_enter_row_yaws, y_type_parking_path[:, 2]])
    path_ks = np.concatenate([drive_to_enter_row_ks, y_type_parking_path[:, 3]])
    dirs = np.concatenate([drive_to_enter_row_dirs, y_type_parking_path[:, 4]])

    return error_code, a_star_node_couts, path_xs, path_ys, path_yaws, path_ks, dirs


def get_backward_steer_dir_for_y_type_parking(start_pose, end_pose):
    # right turn
    if end_pose[1] - start_pose[1] > 0:
        backward_steer = np.sign(1 * math.cos(start_pose[2]))
    # left turn
    else:
        backward_steer = np.sign(-1 * math.cos(start_pose[2]))

    return backward_steer


# parking path is planned inversely from end pose
def search_y_type_parking_path(
    car_model: CarModel,
    config_env: OrchardGeometryEnvironment,
    end_pose,
    backward_steer_dir,
    forward_steer_dir,
    max_steer_backward=0.4,
    max_steer_forward=0.45,
    max_backward_distance=3.5,
    max_forward_distance=2.0,
    min_forward_distance=1.4,
    min_backward_distance=0.7,
    min_steer_backward=0.3,
    min_steer_forward=0.3,
    step_size=0.1,
    debug=False,
):
    # check end pose
    if not config_env.check_path_feasibility(car_model, np.array([end_pose])):
        print(" [Y-type Planner] The end pose is interfered with the environment!")
        return [], []

    steer_backwards = list(np.arange(min_steer_backward, max_steer_backward + 0.1, 0.1))
    if np.max(steer_backwards) < max_steer_backward:
        steer_backwards.append(max_steer_backward)
    steer_forwards = list(np.arange(min_steer_forward, max_steer_forward + 0.1, 0.1))
    if np.max(steer_forwards) < max_steer_forward:
        steer_forwards.append(max_steer_forward)

    for backward_length in np.arange(max_backward_distance, min_backward_distance, -0.1):
        for forward_length in np.arange(max_forward_distance, min_forward_distance, -0.1):
            for steer_backward in steer_backwards:
                for steer_forward in steer_forwards:
                    human_type_path_in_baselink = get_y_type_parking_path(
                        car_model,
                        backward_length,
                        steer_backward * backward_steer_dir,
                        forward_length,
                        steer_forward * forward_steer_dir,
                        step_size,
                    )
                    odom_T_baselink = trans.states2SE3(
                        [end_pose[0], end_pose[1], 0, 0, 0, end_pose[2]]
                    )
                    y_type_path_in_odom = get_path_in_odom(
                        odom_T_baselink, human_type_path_in_baselink
                    )
                    if config_env.check_path_feasibility(car_model, y_type_path_in_odom):
                        print(
                            "backward distance:%.2f, forward distance:%.2f, backward steer:%.2f, forward steer:%.2f,  "
                            % (
                                backward_length,
                                forward_length,
                                steer_backward,
                                steer_forward,
                            )
                        )
                        if debug:
                            return y_type_path_in_odom, [
                                backward_length,
                                forward_length,
                                steer_backward,
                                steer_forward,
                            ]
                        else:
                            return y_type_path_in_odom
    if debug:
        return [], []
    else:
        return []


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


def get_feasible_path_through_backward(
    car: CarModel,
    config_env: OrchardGeometryEnvironment,
    end_pose,
    backward_steer_dir,
    forward_steer_dir,
    steer_backward,
    forward_distance,
    backward_distance,
    steer_forward,
    step_size,
):
    path_is_feasible = False
    while True:
        y_type_path_in_baselink = get_y_type_parking_path(
            car,
            backward_distance,
            steer_backward * backward_steer_dir,
            forward_distance,
            steer_forward * forward_steer_dir,
            step_size,
        )
        odom_T_baselink = trans.states2SE3(
            [end_pose[0], end_pose[1], 0, 0, 0, end_pose[2]]
        )
        y_type_path_in_odom = get_path_in_odom(odom_T_baselink, y_type_path_in_baselink)
        path_is_feasible = config_env.check_path_feasibility(
            car, y_type_path_in_odom, boundary_check=False
        )
        if path_is_feasible:
            break
        backward_distance += 0.1

    return y_type_path_in_odom, backward_distance


def get_warm_start_path_y_type(
    car: CarModel,
    config_env: OrchardGeometryEnvironment,
    start_pose,
    end_pose,
    backward_steer_dir,
    forward_steer_dir,
    steer_backward,
    forward_distance,
    backward_distance,
    steer_forward,
    step_size,
    drive_row_offset=4.5,
):
    y_type_parking_path, _ = get_feasible_path_through_backward(
        car,
        config_env,
        end_pose,
        backward_steer_dir,
        forward_steer_dir,
        steer_backward,
        forward_distance,
        backward_distance,
        steer_forward,
        step_size,
    )
    # forward path
    intermediate_pose = y_type_parking_path[0][:3]

    # get topological waypoints for headland traveling
    way_points = config_env.get_topology_waypoints(
        start_pose, intermediate_pose, drive_row_offset=drive_row_offset
    )

    search_heuristic = ReferenceLineHeuristic(way_points, intermediate_pose, car)

    planner_to_intermediate_pose = HybridAStarSearch(
        start_pose,
        intermediate_pose,
        config_env,
        car,
        search_heuristic,
        motion_type="Pawn",
        plan_resolution=step_size,
    )
    (
        drive_to_enter_row_xs,
        drive_to_enter_row_ys,
        drive_to_enter_row_yaws,
        drive_to_enter_row_dirs,
        _,
        a_star_node_couts,
    ) = planner_to_intermediate_pose.hybrid_a_star_search(max_nodes=400)

    if len(drive_to_enter_row_xs) == 0:
        print("cannot search a path to parking start pose!!")
        error_code = ERROR_CODE_FOR_HYBRID_A_STAR
        return error_code, a_star_node_couts, [], [], [], [], []
    (
        drive_to_enter_row_xs,
        drive_to_enter_row_ys,
        drive_to_enter_row_yaws,
        drive_to_enter_row_ks,
        _,
    ) = calc_spline_course(drive_to_enter_row_xs, drive_to_enter_row_ys, ds=step_size)
    drive_to_enter_row_dirs = np.ones_like(drive_to_enter_row_xs)
    path_xs = np.concatenate([drive_to_enter_row_xs, y_type_parking_path[:, 0]])
    path_ys = np.concatenate([drive_to_enter_row_ys, y_type_parking_path[:, 1]])
    path_yaws = np.concatenate([drive_to_enter_row_yaws, y_type_parking_path[:, 2]])
    path_ks = np.concatenate([drive_to_enter_row_ks, y_type_parking_path[:, 3]])
    dirs = np.concatenate([drive_to_enter_row_dirs, y_type_parking_path[:, 4]])

    return error_code, a_star_node_couts, path_xs, path_ys, path_yaws, path_ks, dirs
