import numpy as np
import transformation as trans
from shapely import box
import math
from utils.cubic_spline import calc_spline_course
from utils.navigation_utils import get_dubins_path
import utils.reeds_shepp as rs_curves
from orchard_geometry_environment import OrchardGeometryEnvironment
from car_model import CarModel
from shapely.geometry import LineString
import dubins

NEAR_SIDE = 1
FAR_SIDE = 2
LEAVE_POSE = 1
ENTER_POSE = 2


def filter_consecutive_duplicate(array):
    return np.array(
        [elem for i, elem in enumerate(array) if (elem - array[i - 1]).any()]
    )


def get_dubins_turn_dirs(start_pose, end_pose, turning_radius):
    path = dubins.shortest_path(start_pose, end_pose, turning_radius)
    configs, _ = path.sample_many(0.1)
    first_config = configs[0]
    last_config = configs[-1]

    start_turn_dir = -1 if first_config[2] > start_pose[2] else 1
    end_turn_dir = -1 if last_config[2] > end_pose[2] else 1

    return start_turn_dir, end_turn_dir


def get_steer_dir_for_enter_calculation(start_pose, end_pose):
    # right turn
    if end_pose[1] - start_pose[1] > 0:
        backward_steer = np.sign(1 * math.cos(start_pose[2]))
    # left turn
    else:
        backward_steer = np.sign(-1 * math.cos(start_pose[2]))

    return backward_steer


def get_offset_poses_for_row_traversing(
    start_leave_pose_base,
    end_enter_pose_base,
    car,
    config_env,
    max_steer_angle=0.55,
    plt=None,
):
    turn_dir = get_steer_dir_for_enter_calculation(
        start_leave_pose_base, end_enter_pose_base
    )

    leave_dist_offset, start_leave_pose_offset, leave_path = get_offset_pose(
        start_leave_pose_base,
        LEAVE_POSE,
        turn_dir,
        car,
        config_env,
        steer_angle=max_steer_angle,
    )
    print("backward distance for leaving is %.2f" % (leave_dist_offset))

    enter_dist_offset, end_enter_pose_offset, enter_path = get_offset_pose(
        end_enter_pose_base,
        ENTER_POSE,
        turn_dir,
        car,
        config_env,
        steer_angle=max_steer_angle,
    )
    print("backward distance for entering is %.2f" % (enter_dist_offset))

    if plt is not None:
        plt.plot(leave_path[:, 0], leave_path[:, 1])
        plt.plot(enter_path[:, 0], enter_path[:, 1])

    return start_leave_pose_offset, end_enter_pose_offset


def get_car_outmost_pose(
    tree_rows, row_id, base_yaw, car: CarModel, side=NEAR_SIDE, pose_type=ENTER_POSE
):
    lower_tree_row = tree_rows[row_id, :, :]
    upper_tree_row = tree_rows[row_id + 1, :, :]

    if side == NEAR_SIDE:
        lower_tree_row_end = lower_tree_row[0, :]
        upper_tree_row_end = upper_tree_row[0, :]
        tree_row_outmost_x = min(lower_tree_row_end[0], upper_tree_row_end[0])
        # Assume the car is fully pulled out of the row
        vehicle_body_bounds = car.car_poly.bounds
        min_x, max_x = vehicle_body_bounds[0], vehicle_body_bounds[2]
        if len(car.aux_polys) > 0:
            for aux_poly in car.aux_polys:
                aux_bounds = aux_poly.bounds
                min_x = min(min_x, aux_bounds[0])
                max_x = max(max_x, aux_bounds[2])

        car_offset_x = abs(min_x) if pose_type == LEAVE_POSE else abs(max_x)
        car_outmost_pos = np.array(
            [
                tree_row_outmost_x - car_offset_x,
                (lower_tree_row_end[1] + upper_tree_row_end[1]) / 2,
                base_yaw,
            ]
        )
    else:
        lower_tree_row_end = lower_tree_row[1, :]
        upper_tree_row_end = upper_tree_row[1, :]
        tree_row_outmost_x = max(lower_tree_row_end[0], upper_tree_row_end[0])
        car_offset_x = (
            car.AXLE_TO_BACK if pose_type == LEAVE_POSE else car.AXLE_TO_FRONT
        )
        car_outmost_pos = np.array(
            [
                tree_row_outmost_x + car_offset_x,
                (lower_tree_row_end[1] + upper_tree_row_end[1]) / 2,
                base_yaw,
            ]
        )

    return car_outmost_pos


def get_start_end_pose_for_dubins(
    map_tree_rows,
    start_row_id,
    end_row_id,
    car,
    config_env,
    max_steer_angle=0.55,
    plt=None,
    side=NEAR_SIDE,
    extra_offset_enter_dist=0.0,  # outer direction of the row entering
    extra_offset_leave_dist=0.0,  # outer direction of the row leaving
):
    start_leave_pose_base = get_base_pose(
        start_row_id,
        map_tree_rows,
        min_offset=extra_offset_leave_dist,
        side=side,
        pose_type=LEAVE_POSE,
    )

    end_enter_pose_base = get_base_pose(
        end_row_id,
        map_tree_rows,
        min_offset=extra_offset_enter_dist,
        side=side,
        pose_type=ENTER_POSE,
    )

    turn_dir = get_steer_dir_for_enter_calculation(
        start_leave_pose_base, end_enter_pose_base
    )
    leave_turn_dir = turn_dir  # initial dir guess
    enter_turn_dir = turn_dir  # initial dir guess

    while True:
        leave_offset, start_leave_pose_offset, leave_path = get_offset_pose(
            start_leave_pose_base,
            LEAVE_POSE,
            leave_turn_dir,
            car,
            config_env,
            steer_angle=max_steer_angle,
        )

        enter_offset, end_enter_pose_offset, enter_path = get_offset_pose(
            end_enter_pose_base,
            ENTER_POSE,
            enter_turn_dir,
            car,
            config_env,
            steer_angle=max_steer_angle,
        )

        start_turn_dir, end_turn_dir = get_dubins_turn_dirs(
            start_leave_pose_offset, end_enter_pose_offset, 1.0 / car.curvature
        )
        if start_turn_dir == leave_turn_dir and end_turn_dir == enter_turn_dir:
            break
        else:
            leave_turn_dir = start_turn_dir
            enter_turn_dir = end_turn_dir

    if plt is not None:
        plt.plot(leave_path[:, 0], leave_path[:, 1])
        plt.plot(enter_path[:, 0], enter_path[:, 1])

    return start_leave_pose_offset, end_enter_pose_offset, leave_offset, enter_offset


def get_base_pose(
    row_id, map_tree_rows, min_offset, side=NEAR_SIDE, pose_type=LEAVE_POSE
):
    near_side_end = [
        (map_tree_rows[row_id, 0, 0] + map_tree_rows[row_id + 1, 0, 0]) / 2,
        (map_tree_rows[row_id, 0, 1] + map_tree_rows[row_id + 1, 0, 1]) / 2,
    ]
    far_side_end = [
        (map_tree_rows[row_id, 1, 0] + map_tree_rows[row_id + 1, 1, 0]) / 2,
        (map_tree_rows[row_id, 1, 1] + map_tree_rows[row_id + 1, 1, 1]) / 2,
    ]

    row_yaw = np.arctan2(
        far_side_end[1] - near_side_end[1], far_side_end[0] - near_side_end[0]
    )

    if pose_type == LEAVE_POSE:
        if side == FAR_SIDE:
            pose_yaw = row_yaw
        if side == NEAR_SIDE:
            pose_yaw = row_yaw + np.pi
        extend_dir = 1

    if pose_type == ENTER_POSE:
        if side == FAR_SIDE:
            pose_yaw = row_yaw + np.pi
        if side == NEAR_SIDE:
            pose_yaw = row_yaw
        extend_dir = -1

    if side == NEAR_SIDE:
        pose_position = (
            near_side_end
            + np.array([np.cos(pose_yaw), np.sin(pose_yaw)]) * min_offset * extend_dir
        )

    if side == FAR_SIDE:
        pose_position = (
            far_side_end
            + np.array([np.cos(pose_yaw), np.sin(pose_yaw)]) * min_offset * extend_dir
        )

    base_pose = np.array([pose_position[0], pose_position[1], pose_yaw])

    return base_pose


def get_offset_pose(
    init_pose,
    pose_type,
    turn_out_dir,
    car,
    config_env,
    steer_angle=0.55,
    delta_yaw=math.radians(45),
    max_offset=5,
    accuracy=0.1,
):
    path_is_feasible = False
    pose = np.copy(init_pose)
    init_x, init_y, init_yaw = init_pose[0], init_pose[1], init_pose[2]

    motion_dir = -1 if pose_type == ENTER_POSE else 1
    offset_dir = -1 if pose_type == ENTER_POSE else 1
    # while True:
    for dist in np.arange(0, max_offset + accuracy, accuracy):
        x = init_x + dist * np.cos(init_yaw) * offset_dir
        y = init_y + dist * np.sin(init_yaw) * offset_dir
        pose = np.array([x, y, init_yaw])
        # print("x: ", x)
        path = car.calculate_motion_path(
            pose, [steer_angle * turn_out_dir, motion_dir], delta_yaw, accuracy
        )
        path_is_feasible = config_env.check_path_feasibility(
            car, path, boundary_check=False
        )
        if path_is_feasible:
            break

    if dist >= max_offset:
        print("no solution is available!! dist: %.2f, x: %.2f" % (dist, x))

    return dist, pose, path


def get_dubins_path_full(pose_start, pose_end, turning_radius, step_size=0.1):
    configurations = get_dubins_path(
        pose_start, pose_end, turning_radius, step_size=step_size
    )
    # find the line part of the path
    rx, ry, ryaw, rk, rs = calc_spline_course(
        configurations[:, 0], configurations[:, 1]
    )
    dirs = np.ones_like(rx)

    path = np.vstack([rx, ry, ryaw, rk, dirs]).T
    return path


def get_start_end_pose_for_reeds_shepp(
    map_tree_rows,
    start_row_id,
    end_row_id,
    car,
    config_env,
    max_steer_angle=0.55,
    plt=None,
    side=NEAR_SIDE,
    extra_offset_enter_dist=0.0,  # outer direction of the row entering
    extra_offset_leave_dist=0.0,  # outer direction of the row leaving
):
    start_leave_pose_base = get_base_pose(
        start_row_id,
        map_tree_rows,
        min_offset=extra_offset_leave_dist,
        side=side,
        pose_type=LEAVE_POSE,
    )

    end_enter_pose_base = get_base_pose(
        end_row_id,
        map_tree_rows,
        min_offset=extra_offset_enter_dist,
        side=side,
        pose_type=ENTER_POSE,
    )

    turn_dir = get_steer_dir_for_enter_calculation(
        start_leave_pose_base, end_enter_pose_base
    )

    leave_offset, start_leave_pose_offset, leave_path = get_offset_pose(
        start_leave_pose_base,
        LEAVE_POSE,
        turn_dir,
        car,
        config_env,
        steer_angle=max_steer_angle,
    )

    enter_offset, end_enter_pose_offset, enter_path = get_offset_pose(
        end_enter_pose_base,
        ENTER_POSE,
        turn_dir,
        car,
        config_env,
        steer_angle=max_steer_angle,
    )

    if side == NEAR_SIDE:
        outmost_x = np.min([start_leave_pose_base[0], end_enter_pose_offset[0]])
    else:
        outmost_x = np.max([start_leave_pose_base[0], end_enter_pose_offset[0]])

    start_leave_pose_offset[0] = outmost_x
    end_enter_pose_offset[0] = outmost_x

    leave_offset = abs(outmost_x - start_leave_pose_base[0])
    enter_offset = abs(outmost_x - end_enter_pose_base[0])
    if plt is not None:
        plt.plot(leave_path[:, 0], leave_path[:, 1])
        plt.plot(enter_path[:, 0], enter_path[:, 1])

    return start_leave_pose_offset, end_enter_pose_offset, leave_offset, enter_offset


def get_all_reeds_shepp_paths_full(pose_start, pose_end, turning_radius, step_size=0.1):
    all_reeds_shepp_paths = rs_curves.calc_all_paths(
        pose_start[0],
        pose_start[1],
        pose_start[2],
        pose_end[0],
        pose_end[1],
        pose_end[2],
        1.0 / turning_radius,
        step_size,
    )

    results = []
    for reeds_shepp_path in all_reeds_shepp_paths:
        path = np.array(
            [
                reeds_shepp_path.x,
                reeds_shepp_path.y,
                reeds_shepp_path.yaw,
                reeds_shepp_path.cs,
                reeds_shepp_path.directions,
            ]
        ).T
        results.append(path)

    return results


def get_circle_back_path_full(
    pose_start, pose_end, turning_radius, car: CarModel, side=NEAR_SIDE, step_size=0.1
):
    w = abs(pose_start[1] - pose_end[1])
    turning_radius = max(turning_radius, 1.0 / car.curvature)
    if w >= turning_radius * 2:
        # print("w is larger than 2R, use dubins path instead")
        return get_dubins_path_full(pose_start, pose_end, turning_radius, step_size)

    Rf = turning_radius
    Rb = turning_radius
    while True:
        theta = np.pi / 2 + np.arcsin((Rb + w - Rf) / (Rf + Rb))
        if side == NEAR_SIDE:
            if (
                pose_start[0] - (Rf + Rb) * np.cos(theta - np.pi / 2)
                < pose_end[0] - step_size
            ):
                break
        if side == FAR_SIDE:
            if (
                pose_start[0] + (Rf + Rb) * np.cos(theta - np.pi / 2)
                > pose_end[0] + step_size
            ):
                break
        Rf *= 1.05
        Rb *= 1.05

    turn_dir = get_steer_dir_for_enter_calculation(pose_start, pose_end)

    # forward path
    forward_path = car.calculate_motion_path_new(
        init_pose=pose_start,
        motion_dir=1,
        steer_dir=turn_dir,
        turning_radius=Rf,
        delta_yaw=theta,
        step_size=step_size,
    )

    # back path
    back_path = car.calculate_motion_path_new(
        init_pose=forward_path[-1, :3],
        motion_dir=-1,
        steer_dir=-turn_dir,
        turning_radius=Rb,
        delta_yaw=np.pi - theta,
        step_size=step_size,
    )

    # straight path

    straight_path = get_dubins_path_full(
        back_path[-1, :3], pose_end, turning_radius, step_size
    )

    # combine paths
    path = np.vstack([forward_path, back_path, straight_path])

    return path


def sample_start_end_pose_for_dubins(
    map_tree_rows,
    start_row_id,
    end_row_id,
    car: CarModel,
    config_env: OrchardGeometryEnvironment,
    max_steer_angle=0.55,
    plt=None,
    side=NEAR_SIDE,
    accuracy=0.2,  # accuracy of the offset sampling
    extra_offset_enter_dist=0.0,  # outer direction of the row entering
    extra_offset_leave_dist=0.0,  # outer direction of the row leaving
):
    start_leave_pose_base = get_base_pose(
        start_row_id,
        map_tree_rows,
        min_offset=extra_offset_leave_dist,
        side=side,
        pose_type=LEAVE_POSE,
    )

    end_enter_pose_base = get_base_pose(
        end_row_id,
        map_tree_rows,
        min_offset=extra_offset_enter_dist,
        side=side,
        pose_type=ENTER_POSE,
    )

    start_leave_outmost_pose = get_car_outmost_pose(
        map_tree_rows,
        start_row_id,
        start_leave_pose_base[2],
        car,
        side=side,
        pose_type=LEAVE_POSE,
    )

    end_enter_outmost_pose = get_car_outmost_pose(
        map_tree_rows,
        end_row_id,
        end_enter_pose_base[2],
        car,
        side=side,
        pose_type=ENTER_POSE,
    )

    if side == NEAR_SIDE:
        outmost_pose_x = min(start_leave_outmost_pose[0], end_enter_outmost_pose[0])
        start_leave_x_samples = np.arange(
            start_leave_pose_base[0], outmost_pose_x - accuracy, -accuracy
        )
        end_enter_x_samples = np.arange(
            end_enter_pose_base[0], outmost_pose_x - accuracy, -accuracy
        )
    else:
        outmost_pose_x = max(start_leave_outmost_pose[0], end_enter_outmost_pose[0])
        start_leave_x_samples = np.arange(
            start_leave_pose_base[0], outmost_pose_x + accuracy, accuracy
        )
        end_enter_x_samples = np.arange(
            end_enter_pose_base[0], outmost_pose_x + accuracy, accuracy
        )

    results = None
    current_min_distance_to_boundary = -np.inf

    for start_leave_x in start_leave_x_samples:
        for end_enter_x in end_enter_x_samples:
            safe_start_pose = np.array(
                [start_leave_x, start_leave_pose_base[1], start_leave_pose_base[2]]
            )
            safe_end_pose = np.array(
                [end_enter_x, end_enter_pose_base[1], end_enter_pose_base[2]]
            )
            turning_path = get_dubins_path_full(
                safe_start_pose, safe_end_pose, 1.0 / car.curvature
            )

            if turning_path is None:
                continue
            if config_env.check_path_feasibility(
                car, turning_path[:, :3], boundary_check=False, aux_check=True
            ):
                min_distance_to_boundary = config_env.get_min_distance_to_boundary(
                    car, turning_path[:, :3], with_aux=True
                )
                if min_distance_to_boundary > current_min_distance_to_boundary:
                    current_min_distance_to_boundary = min_distance_to_boundary
                    leave_offset = abs(start_leave_x - start_leave_pose_base[0])
                    enter_offset = abs(end_enter_x - end_enter_pose_base[0])
                    results = (
                        safe_start_pose,
                        safe_end_pose,
                        leave_offset,
                        enter_offset,
                    )

    return results


def sample_start_end_pose_for_reeds_shepp(
    map_tree_rows,
    start_row_id,
    end_row_id,
    car: CarModel,
    config_env: OrchardGeometryEnvironment,
    max_steer_angle=0.55,
    plt=None,
    side=NEAR_SIDE,
    accuracy=0.2,  # accuracy of the offset sampling
    extra_offset_enter_dist=0.0,  # outer direction of the row entering
    extra_offset_leave_dist=0.0,  # outer direction of the row leaving
):
    start_leave_pose_base = get_base_pose(
        start_row_id,
        map_tree_rows,
        min_offset=extra_offset_leave_dist,
        side=side,
        pose_type=LEAVE_POSE,
    )

    end_enter_pose_base = get_base_pose(
        end_row_id,
        map_tree_rows,
        min_offset=extra_offset_enter_dist,
        side=side,
        pose_type=ENTER_POSE,
    )

    start_leave_outmost_pose = get_car_outmost_pose(
        map_tree_rows,
        start_row_id,
        start_leave_pose_base[2],
        car,
        side=side,
        pose_type=LEAVE_POSE,
    )

    end_enter_outmost_pose = get_car_outmost_pose(
        map_tree_rows,
        end_row_id,
        end_enter_pose_base[2],
        car,
        side=side,
        pose_type=ENTER_POSE,
    )

    if side == NEAR_SIDE:
        outmost_pose_x = min(start_leave_outmost_pose[0], end_enter_outmost_pose[0])
        start_leave_x_samples = np.arange(
            start_leave_pose_base[0], outmost_pose_x - accuracy, -accuracy
        )
        end_enter_x_samples = np.arange(
            end_enter_pose_base[0], outmost_pose_x - accuracy, -accuracy
        )
    else:
        outmost_pose_x = max(start_leave_outmost_pose[0], end_enter_outmost_pose[0])
        start_leave_x_samples = np.arange(
            start_leave_pose_base[0], outmost_pose_x + accuracy, accuracy
        )
        end_enter_x_samples = np.arange(
            end_enter_pose_base[0], outmost_pose_x + accuracy, accuracy
        )

    results = None
    current_min_distance_to_boundary = -np.inf

    for start_leave_x in start_leave_x_samples:
        for end_enter_x in end_enter_x_samples:
            safe_start_pose = np.array(
                [start_leave_x, start_leave_pose_base[1], start_leave_pose_base[2]]
            )
            safe_end_pose = np.array(
                [end_enter_x, end_enter_pose_base[1], end_enter_pose_base[2]]
            )
            all_turning_paths = get_all_reeds_shepp_paths_full(
                safe_start_pose, safe_end_pose, 1.0 / car.curvature
            )
            min_distance_to_boundary = -np.inf

            if len(all_turning_paths) == 0:
                continue
            for path in all_turning_paths:
                if config_env.check_path_feasibility(
                    car, path[:, :3], boundary_check=False, aux_check=True
                ):
                    distance = config_env.get_min_distance_to_boundary(
                        car, path[:, :3], with_aux=True
                    )
                    if distance > min_distance_to_boundary:
                        min_distance_to_boundary = distance

            if min_distance_to_boundary > current_min_distance_to_boundary:
                current_min_distance_to_boundary = min_distance_to_boundary
                leave_offset = abs(start_leave_x - start_leave_pose_base[0])
                enter_offset = abs(end_enter_x - end_enter_pose_base[0])
                results = (safe_start_pose, safe_end_pose, leave_offset, enter_offset)

    return results


def get_start_end_pose(
    map_tree_rows,
    start_row_id,
    end_row_id,
    side=NEAR_SIDE,
    extra_offset_enter_dist=0.4,  # outer direction of the row entering
    extra_offset_leave_dist=1.1,  # outer direction of the row leaving
):
    N, M = start_row_id, end_row_id
    # get near side start and end pose
    near_side_start = [
        (map_tree_rows[N, 0, 0] + map_tree_rows[N + 1, 0, 0]) / 2,
        (map_tree_rows[N, 0, 1] + map_tree_rows[N + 1, 0, 1]) / 2,
    ]
    far_side_start = [
        (map_tree_rows[N, 1, 0] + map_tree_rows[N + 1, 1, 0]) / 2,
        (map_tree_rows[N, 1, 1] + map_tree_rows[N + 1, 1, 1]) / 2,
    ]

    near_side_end = [
        (map_tree_rows[M, 0, 0] + map_tree_rows[M + 1, 0, 0]) / 2,
        (map_tree_rows[M, 0, 1] + map_tree_rows[M + 1, 0, 1]) / 2,
    ]
    far_side_end = [
        (map_tree_rows[M, 1, 0] + map_tree_rows[M + 1, 1, 0]) / 2,
        (map_tree_rows[M, 1, 1] + map_tree_rows[M + 1, 1, 1]) / 2,
    ]

    start_row_yaw = np.arctan2(
        far_side_start[1] - near_side_start[1], far_side_start[0] - near_side_start[0]
    )
    end_row_yaw = np.arctan2(
        far_side_end[1] - near_side_end[1], far_side_end[0] - near_side_end[0]
    )

    if side == NEAR_SIDE:
        start_pose_x = min(map_tree_rows[N, 0, 0], map_tree_rows[N + 1, 0, 0])
        start_pose_y = (map_tree_rows[N, 0, 1] + map_tree_rows[N + 1, 0, 1]) / 2

        start_pose = [
            start_pose_x - extra_offset_leave_dist * np.cos(start_row_yaw),
            start_pose_y - extra_offset_leave_dist * np.sin(start_row_yaw),
            np.pi + start_row_yaw,  # exit on near side
        ]

        end_pose_x = min(map_tree_rows[M, 0, 0], map_tree_rows[M + 1, 0, 0])
        end_pose_y = (map_tree_rows[M, 0, 1] + map_tree_rows[M + 1, 0, 1]) / 2
        end_pose = [
            end_pose_x - extra_offset_enter_dist * np.cos(end_row_yaw),
            end_pose_y - extra_offset_enter_dist * np.sin(end_row_yaw),
            end_row_yaw,
        ]

    if side == FAR_SIDE:
        start_pose_x = max(map_tree_rows[N, 1, 0], map_tree_rows[N + 1, 1, 0])
        start_pose_y = (map_tree_rows[N, 1, 1] + map_tree_rows[N + 1, 1, 1]) / 2

        start_pose = [
            start_pose_x + extra_offset_leave_dist * np.cos(start_row_yaw),
            start_pose_y + extra_offset_leave_dist * np.sin(start_row_yaw),
            start_row_yaw,  # exit on near side
        ]

        end_pose_x = max(map_tree_rows[M, 1, 0], map_tree_rows[M + 1, 1, 0])
        end_pose_y = (map_tree_rows[M, 1, 1] + map_tree_rows[M + 1, 1, 1]) / 2
        end_pose = [
            end_pose_x + extra_offset_enter_dist * np.cos(end_row_yaw),
            end_pose_y + extra_offset_enter_dist * np.sin(end_row_yaw),
            end_row_yaw + np.pi,
        ]

    return start_pose, end_pose


def sample_start_end_pose_for_circle_back(
    map_tree_rows,
    start_row_id,
    end_row_id,
    car: CarModel,
    config_env: OrchardGeometryEnvironment,
    max_steer_angle=0.55,
    plt=None,
    side=NEAR_SIDE,
    accuracy=0.2,  # accuracy of the offset sampling
    extra_offset_enter_dist=0.0,  # outer direction of the row entering
    extra_offset_leave_dist=0.0,  # outer direction of the row leaving
):
    # start_leave_pose_base = get_base_pose(
    #     start_row_id,
    #     map_tree_rows,
    #     min_offset=extra_offset_leave_dist,
    #     side=side,
    #     pose_type=LEAVE_POSE,
    # )

    # end_enter_pose_base = get_base_pose(
    #     end_row_id,
    #     map_tree_rows,
    #     min_offset=extra_offset_enter_dist,
    #     side=side,
    #     pose_type=ENTER_POSE,
    # )
    start_leave_pose_base, end_enter_pose_base = get_start_end_pose(
        map_tree_rows,
        start_row_id,
        end_row_id,
        side,
        extra_offset_enter_dist,
        extra_offset_leave_dist,
    )

    if side == NEAR_SIDE:
        offset_step = -accuracy
    else:
        offset_step = accuracy

    current_offset = 0.0
    while abs(current_offset) < 5:
        start_leave_pose_offset = start_leave_pose_base + np.array(
            [current_offset, 0, 0]
        )
        path = get_circle_back_path_full(
            start_leave_pose_offset, end_enter_pose_base, 1.0 / car.curvature, car, side
        )
        if config_env.check_path_feasibility(
            car, path[:, :3], boundary_check=False, aux_check=True
        ):
            break
        current_offset += offset_step
        # print("current_offset: ", current_offset)

    return (start_leave_pose_offset, end_enter_pose_base, abs(current_offset), 0.0)


def classic_circle_back_turning_path(
    start_exit_pose,
    end_enter_pose,
    map_env: OrchardGeometryEnvironment,
    car_model: CarModel,
    step_size=0.1,
    plt=None,
):
    # get all reeds shepp paths
    to_goal_paths = rs_curves.calc_all_paths(
        start_exit_pose[0],
        start_exit_pose[1],
        start_exit_pose[2],
        end_enter_pose[0],
        end_enter_pose[1],
        end_enter_pose[2],
        car_model.curvature,
        step_size,
    )

    feasible_paths = []
    for path in to_goal_paths:
        path_point = np.array([path.x, path.y]).T
        path_line = LineString(path_point).buffer(0.3, cap_style=2, join_style=1)
        nearest_poly = map_env.get_nearest_poly_of_a_geometry(path_line)
        if nearest_poly.intersects(path_line):
            continue
        else:
            # plt.fill(*path_line.exterior.xy, color="red", alpha=0.1)
            # plt.plot(path.x, path.y, color='red')
            feasible_paths.append(path)

    if len(feasible_paths) > 0:
        optimal_path = None
        optimal_cost = 99999
        for path in feasible_paths:
            path_lengths = np.array(path.lengths)
            backward_lengths_idxs = np.where(path_lengths < 0)
            backward_lengths = path_lengths[backward_lengths_idxs]
            path_cost = np.abs(backward_lengths.sum())
            if path_cost < optimal_cost:
                optimal_path = path
                optimal_cost = path_cost
    else:
        optimal_path = feasible_paths[0]

    # empty_car.draw_car(plt, traj[0,0],traj[0,1],traj[0,2],alpha=1,color='red')
    # empty_car.draw_car(plt, traj[-1,0],traj[-1,1],traj[-1,2],alpha=1,color='red')

    turning_path = np.array(
        [
            optimal_path.x,
            optimal_path.y,
            optimal_path.yaw,
            optimal_path.cs,
            optimal_path.directions,
        ]
    ).T
    # move the whole path until it is not interfere with the rows
    path_is_feasible = map_env.check_path_feasibility(
        car_model, turning_path[:, :3], boundary_check=False
    )
    offset = 0
    while not path_is_feasible:
        path_is_feasible = map_env.check_path_feasibility(
            car_model, turning_path[:, :3], boundary_check=False
        )
        if plt is not None:
            plt.plot(turning_path[:, 0], turning_path[:, 1], "--", c="black", alpha=0.5)
        offset += step_size
        turning_path[:, :2] += np.array(
            [
                [
                    step_size * np.cos(start_exit_pose[2]),
                    step_size * np.sin(start_exit_pose[2]),
                ]
            ]
        )

    if offset > step_size:
        leave_path = get_dubins_path_full(
            start_exit_pose, turning_path[0, :3], 1.0 / car_model.curvature
        )
        turning_path = np.vstack([leave_path[:-1], turning_path])
        enter_path = get_dubins_path_full(
            turning_path[-1, :3], end_enter_pose, 1.0 / car_model.curvature
        )
        turning_path = np.vstack([turning_path, enter_path])

    return turning_path
