import numpy as np
import transformation as trans
from shapely import box
import math
import pickle
import os

NEAR_SIDE = 1
FAR_SIDE = 2
LEAVE_POSE = 1
ENTER_POSE = 2


def plot_arrow(
    x, y, yaw, plt=None, length=1.0, width=0.5, fc="r", ec="k"
):  # pragma: no cover
    """Plot arrow."""
    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plt.arrow(
                ix,
                iy,
                length * np.cos(iyaw),
                length * np.sin(iyaw),
                fc=fc,
                ec=ec,
                head_width=width,
                head_length=width,
            )
            plt.plot(x, y)
    else:
        plt.arrow(
            x,
            y,
            length * np.cos(yaw),
            length * np.sin(yaw),
            fc=fc,
            ec=ec,
            head_width=width,
            head_length=width,
        )
        plt.plot(x, y)


def create_tree_rows(row_num, row_width, row_lengths, slope_angle=0, l_std=0.0):
    tree_rows = []
    x = 0
    delta_x = row_width * np.tan(slope_angle)
    for i in range(row_num):
        if isinstance(row_lengths, list) or isinstance(row_lengths, np.ndarray):
            row_length = row_lengths[i]
        else:
            row_length = row_lengths
        y = row_width * i
        x = delta_x * i
        x += np.random.uniform(-l_std, l_std)
        tree_row = np.array([[x, y], [x + row_length, y]])
        tree_rows.append(tree_row)

    tree_rows = np.array(tree_rows)
    return tree_rows


def generate_utm_field_transform(map_origin, map_heading):
    # Generate the transformation from utm to odom frame
    T = trans.states2SE3([-map_origin[0], -map_origin[1], 0, 0, 0, 0])
    R = trans.states2SE3([0, 0, 0, 0, 0, -map_heading])
    return R.dot(T)


def load_map(map_file):
    # get static map information
    map_dict = eval(open(map_file, "r").read())
    map_heading = map_dict["heading"]
    map_origin = np.asarray(map_dict["origin"])
    odom_T_utm = generate_utm_field_transform(map_origin, map_heading)
    map_tree_rows = np.asarray(map_dict["tree_rows"])
    map_headland_width = np.asarray(map_dict["headland_width"])
    obstacles = []
    contour_points = []
    if "obstacles" in map_dict.keys():
        obstacles = map_dict["obstacles"]
    if "headland_contour" in map_dict.keys():
        contour_points = np.asarray(map_dict["headland_contour"])

    return map_tree_rows, odom_T_utm, obstacles, contour_points


def get_steer_dir_for_enter_calculation(start_pose, end_pose):
    # right turn
    if end_pose[1] - start_pose[1] > 0:
        backward_steer = np.sign(1 * math.cos(start_pose[2]))
    # left turn
    else:
        backward_steer = np.sign(-1 * math.cos(start_pose[2]))

    return backward_steer


def get_start_end_pose_for_dubins(
    map_tree_rows,
    start_row_id,
    end_row_id,
    car,
    config_env,
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

    _, start_leave_pose_offset, leave_path = get_offset_pose(
        start_leave_pose_base,
        LEAVE_POSE,
        turn_dir,
        car,
        config_env,
    )

    _, end_enter_pose_offset, enter_path = get_offset_pose(
        end_enter_pose_base,
        ENTER_POSE,
        turn_dir,
        car,
        config_env,
    )

    if plt is not None:
        plt.plot(leave_path[:, 0], leave_path[:, 1])
        plt.plot(enter_path[:, 0], enter_path[:, 1])

    return start_leave_pose_offset, end_enter_pose_offset


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
    elif side == FAR_SIDE:
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


def get_path_bounds(car, path):
    path_polys = car.get_path_poly(path)
    vehicle_bounds = path_polys[0].bounds
    aux_bounds = []
    for aux_path in path_polys[1]:
        aux_bound = aux_path.bounds
        aux_bounds.append(aux_bound)
    path_bounds = np.array([vehicle_bounds] + aux_bounds)
    path_bound = [
        np.min(path_bounds[:, 0]),
        np.min(path_bounds[:, 1]),
        np.max(path_bounds[:, 2]),
        np.max(path_bounds[:, 3]),
    ]
    box_coords = list(
        box(path_bound[0], path_bound[1], path_bound[2], path_bound[3]).boundary.coords
    )
    return box_coords


def get_path_boundary(car, path):
    path_polys = car.get_path_poly(path)
    vehicle_boundary = np.array(path_polys[0].exterior.coords)
    aux_boundaries = []
    for aux_path in path_polys[1]:
        aux_boundary = np.array(aux_path.exterior.coords)
        aux_boundaries.append(aux_boundary)
    path_boundary = [vehicle_boundary] + aux_boundaries
    path_boundary = np.vstack(path_boundary)
    return path_boundary


def write_dict_into_pkl(path, dict):
    # write the paths into a pkl file
    with open(path, "wb") as f:
        pickle.dump(dict, f)


def load_dict_from_pkl(file_name):
    loaded_dict = {}
    if os.path.isfile(file_name):
        unpickle_file = open(file_name, "rb")
        loaded_dict = pickle.load(unpickle_file)
    else:
        print("reference path file does not exists!")

    return loaded_dict
