import time
import numpy as np
from . import transformation
import math
from skspatial.objects import Line, Point
# from transformations import euler_from_quaternion
import dubins


def swap(var_a, var_b):
    buffer_var = var_a
    var_a = var_b
    var_b = buffer_var
    return var_a, var_b


def decode_map_from_msg(map_msg):
    # decode map message
    map_heading = map_msg.heading
    map_origin = np.asarray([map_msg.origin.x, map_msg.origin.y])
    map_T_utm = generate_utm_field_transform(map_origin, map_heading)
    map_treerows = []
    extra_offset = map_msg.extra_offset
    for tree_row_msg in map_msg.tree_rows:
        map_treerows.append(
            [
                [[tree_row_msg.start_pt.x], [tree_row_msg.start_pt.y]],
                [[tree_row_msg.end_pt.x], [tree_row_msg.end_pt.y]],
            ]
        )
    map_treerows = np.asarray(map_treerows)[:, :, :, 0]

    map_obstacles = []
    for obstacles in map_msg.obstacles:
        map_obstacles.append([obstacles.x, obstacles.y])
    map_obstacles = np.asarray(map_obstacles)

    # print("[decode_map_from_msg] ", map_treerows.shape)
    map_ave_row_width = 5  # TODO: Delete this
    row_centerlins = get_map_centerlines(map_treerows)
    return (
        map_heading,
        map_origin,
        map_T_utm,
        map_treerows,
        map_ave_row_width,
        row_centerlins,
        map_obstacles,
        extra_offset,
    )


def get_map_centerlines(treerows):
    # Get Center line points
    centerlines = []
    for i in range(len(treerows) - 1):
        row_0 = treerows[i]
        row_1 = treerows[i + 1]
        centerline_start = [
            min(row_0[0][0], row_1[0][0]),
            (row_0[0][1] + row_1[0][1]) / 2,
        ]
        centerline_end = [
            min(row_0[1][0], row_1[1][0]),
            (row_0[1][1] + row_1[1][1]) / 2,
        ]
        centerlines.append([centerline_start, centerline_end])
    centerlines = np.asarray(centerlines)
    return centerlines


def generate_utm_field_transform(map_origin, map_heading):
    # Generate the transformation from utm to odom frame
    T = transformation.states2SE3([-map_origin[0], -map_origin[1], 0, 0, 0, 0])
    R = transformation.states2SE3([0, 0, 0, 0, 0, -map_heading])
    return R.dot(T)


def generate_in_row_gnss_backup_path(robot_states, distance=10):
    """
    Generate in row navigation path based on in row gnss history
    """
    step_size = 0.1
    num = int(distance / step_size)
    if robot_states.travel_dir > 0:
        xs = np.linspace(
            robot_states.current_pose[0],
            robot_states.current_pose[0] + distance,
            num=num,
        ).reshape(-1, 1)
        ys = xs * 0 + robot_states.row_center_y_est
        headings = ys * 0
    else:
        xs = np.linspace(
            robot_states.current_pose[0],
            robot_states.current_pose[0] - distance,
            num=num,
        ).reshape(-1, 1)
        ys = xs * 0 + robot_states.row_center_y_est
        headings = ys * 0 + math.pi
    path_trajectory = np.concatenate([xs, ys, headings], axis=1)
    return path_trajectory


def get_projection_point(x_m, y_m, yaw_m, k_m, x, y):

    # get projection point on the curve
    d_vector = np.array([x - x_m, y - y_m])
    tau_vector = np.array([math.cos(yaw_m), math.sin(yaw_m)])
    p_proj = np.array([x_m, y_m]) + (d_vector.dot(tau_vector)) * tau_vector
    yaw_proj = yaw_m + k_m * (d_vector.dot(tau_vector))

    return p_proj, yaw_proj


def generate_in_row_gnss_path(robot_states, start_pt, end_pt, distance=15):
    """
    Generate in row navigation path based on GNSS map
    """
    # calculate heading of row TODO optimize repeated calculation
    row_yaw = math.atan2(
        end_pt[1] - start_pt[1],
        end_pt[0] - start_pt[0],
    )
    # moving direction
    if robot_states.travel_dir < 0:
        row_yaw += math.pi

    start_pt = np.asarray(start_pt).reshape(-1)
    end_pt = np.asarray(end_pt).reshape(-1)

    row_line = Line.from_points(start_pt, end_pt)
    robot_position = Point(robot_states.current_pose[:2])
    # get row projected point
    point_projected = row_line.project_point(robot_position)

    # build the reference path
    step_size = 0.2
    num = int(distance / step_size)
    # create path from current position to a distance
    xs_map = np.linspace(
        point_projected[0],
        point_projected[0] + distance * math.cos(row_yaw),
        num=num,
    ).reshape(-1, 1)

    ys_map = np.linspace(
        point_projected[1],
        point_projected[1] + distance * math.sin(row_yaw),
        num=num,
    ).reshape(-1, 1)

    in_row_path = np.hstack((xs_map, ys_map, row_yaw * np.ones_like(xs_map)))

    return in_row_path


def generate_in_row_gnss_backup_path_camera_frame(
    robot_states, tf_listener, camera_frame_id, distance=10
):
    """
    Generate in row navigation path based on in row gnss history
    This version is optimized by Chen and convert trajectory from odom into camera frame
    """
    step_size = 0.1
    num = int(distance / step_size)
    if robot_states.travel_dir > 0:
        xs = np.linspace(
            robot_states.current_pose[0],
            robot_states.current_pose[0] + distance,
            num=num,
        ).reshape(-1, 1)
        ys = xs * 0 + robot_states.row_center_y_est
        headings = ys * 0
    else:
        xs = np.linspace(
            robot_states.current_pose[0],
            robot_states.current_pose[0] - distance,
            num=num,
        ).reshape(-1, 1)
        ys = xs * 0 + robot_states.row_center_y_est
        headings = ys * 0 + math.pi
    # convert the path into local frame (camera frame)
    camera_T_odom = lookup_transformation(tf_listener, camera_frame_id, "/odom")
    # print("back up xs:", xs.shape)
    xs_camera, ys_camera = convert_2d_xys_to_target_frame(
        xs[:, 0], ys[:, 0], camera_T_odom
    )

    yaw_camera = transformation.SE32states(camera_T_odom)[-1]
    yaws_camera = yaw_camera * np.ones_like(xs_camera)
    path_trajectory = np.vstack([xs_camera, ys_camera, yaws_camera]).T
    return path_trajectory


def convert_2d_xys_to_target_frame(xs_source, ys_source, target_T_source):
    pts_3d = np.vstack((xs_source, ys_source, np.zeros_like(ys_source))).T
    pts_3d_homo = transformation.xyz2homo(pts_3d).T
    points_target_frame_2d = target_T_source.dot(pts_3d_homo)[:2, :].T
    points_target_frame_xs = points_target_frame_2d[:, 0]
    points_target_frame_ys = points_target_frame_2d[:, 1]

    return points_target_frame_xs, points_target_frame_ys


def get_dubins_path(pose_start, pose_end, turning_radius, step_size):
    q0 = (pose_start[0], pose_start[1], pose_start[2])
    q1 = (pose_end[0], pose_end[1], pose_end[2])
    #   print("q0: ", q0)
    #   print("q1: ", q1)
    dubin_path = dubins.shortest_path(q0, q1, turning_radius)
    path, _ = dubin_path.sample_many(step_size)
    path = np.asarray(path)

    return path


def get_straight_path(start_point, end_point, step_size=0.1):

    start_x = start_point[0]
    end_x = end_point[0]

    start_y = start_point[1]
    end_y = end_point[1]

    dx = end_x - start_x
    dy = end_y - start_y
    dist = np.hypot(dx, dy)

    N = int(dist / step_size)

    xs = np.linspace(start_x, end_x, N)
    ys = np.linspace(start_y, end_y, N)
    yaws = np.ones(N) * math.atan2(dy, dx)
    straight_path = np.vstack((xs, ys, yaws)).T

    return straight_path


def get_angle_difference(angle1, angle2):
    vector1 = np.array([math.cos(angle1), math.sin(angle1)])
    vector2 = np.array([math.cos(angle2), math.sin(angle2)])
    dot_product = np.dot(vector1, vector2)
    sin_angle_diff_sign = np.sign(math.sin(angle1 - angle2))
    # sin_angle_diff_sign = math.sin(angle1 - angle2)
    dot_product = min(dot_product, 1)
    dot_product = max(dot_product, -1)
    angle_v1_to_v2 = math.acos(dot_product) * sin_angle_diff_sign
    # print("angle diff: ", angle1 - angle2, "angle_v1_to_v2: ", angle_v1_to_v2)
    return angle_v1_to_v2


def calculate_path_length(xs, ys):
    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.hypot(dx, dy)
    ss = np.cumsum(ds)
    path_length = ss[-1]

    return path_length


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle



def get_2d_pose_of_target_frame(source_pose_in_odom, source_T_target):

    odom_T_source = transformation.states2SE3(
        [
            source_pose_in_odom[0],
            source_pose_in_odom[1],
            0,
            0,
            0,
            source_pose_in_odom[2],
        ]
    )

    odom_T_target = odom_T_source.dot(source_T_target)
    (
        target_x_in_odom,
        target_y_in_odom,
        _,
        _,
        _,
        target_yaw_in_odom,
    ) = transformation.SE32states(odom_T_target)

    return [target_x_in_odom, target_y_in_odom, target_yaw_in_odom]


def angle_wrap(angles):
    # output angle between -pi to pi
    wrap_angle = (angles + math.pi) % (2 * math.pi) - math.pi
    return wrap_angle
