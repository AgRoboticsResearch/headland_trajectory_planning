import math
import numpy as np
from car_model_obca import CarModel
from cubic_spline import calc_spline_course


def wrap_angle(angle):
    """
    Wrap angle between -pi to pi
    """
    wrap_angle = (angle + math.pi) % (2 * math.pi) - math.pi

    return wrap_angle


def process_angle(raw_angles):
    """
    Adjust angels to start from [-pi, pi] and change monotonically
    """
    adjusted_angles = np.zeros_like(raw_angles)
    for i in range(len(adjusted_angles)):
        adjusted_angles[i] = wrap_angle(raw_angles[i])

    adjusted_angles = convert_angle_to_monotonic(adjusted_angles)

    return adjusted_angles


def convert_angle_to_monotonic(raw_angles):
    """
    Convert euler angle including +/- 2pi to 0 jump to continous series data
    """
    if len(raw_angles) <= 1:
        return np.copy(raw_angles)

    angle_monotonic = np.zeros(len(raw_angles))

    angle_monotonic[0] = raw_angles[0]
    for i in range(1, len(raw_angles)):
        difference = raw_angles[i] - raw_angles[i - 1]
        angle_monotonic[i] = angle_monotonic[i - 1] + wrap_angle(difference)

    return angle_monotonic


def get_init_ref_path_coarse(
    car: CarModel, path_xs, path_ys, path_yaws, path_ks, dirs, desired_v=0.5, ds=0.1
):
    segmented_paths = []
    # x, y, v, yaw, steer
    ref_traj = np.vstack([path_xs, path_ys, dirs, path_yaws, path_ks]).T
    ref_traj[:, 2] = dirs * desired_v
    ref_traj[:, -1] = np.arctan(car.WHEEL_BASE * np.array(path_ks))

    ref_traj[:, 3] = process_angle(ref_traj[:, 3])
    ref_traj[0, 2] = 0
    ref_traj[-1, 2] = 0

    return ref_traj


def get_init_ref_path(
    car: CarModel, path_xs, path_ys, path_yaws, path_ks, dirs, desired_v=0.5, ds=0.1
):
    segmented_paths = []
    # x, y, v, yaw, steer
    # ref_traj = np.vstack([path_xs, path_ys, dirs, path_yaws, path_ks]).T
    # ref_traj[:, 2] = dirs * desired_v
    # ref_traj[:, -1] = np.arctan(car.WHEEL_BASE * np.array(path_ks))

    ref_path = np.vstack([path_xs, path_ys, path_yaws, path_ks, dirs]).T
    diffs = np.diff(ref_path[:, -1])
    dividers = np.where(diffs != 0)[0]
    if len(dividers) > 0:
        for i, idx in enumerate(dividers):
            if i == 0:
                segmented_paths.append(ref_path[: idx + 1])
            else:
                segmented_paths.append(ref_path[dividers[i - 1] + 1 : idx + 1])
        segmented_paths.append(ref_path[idx + 1 :])
    else:
        segmented_paths.append(ref_path)

    ref_traj = np.array([])
    for path in segmented_paths:
        xs, ys, yaws, ks, _ = calc_spline_course(path[:, 0], path[:, 1], ds=ds)
        if path[-1, -1] < 0:
            yaws = wrap_angle(np.array(yaws) + np.pi)
            steer_dir = -1
        else:
            steer_dir = 1
        # steers = []
        steers = np.arctan(car.WHEEL_BASE * np.array(ks)) * steer_dir
        # for k in ks:
        #     steer = np.arctan(car.WHEEL_BASE * k) * steer_dir
        #     if abs(steer) > car.MAX_STEER:
        #         steers.append(car.MAX_STEER * np.sign(steer))
        #     else:
        #         steers.append(steer)
        vs = np.ones_like(xs) * path[0, -1] * desired_v
        # initial speed and steer is zero
        vs[0] = 0
        steers[0] = 0
        traj = np.vstack([xs, ys, vs, yaws, steers]).T
        if len(ref_traj) == 0:
            ref_traj = traj
        else:
            ref_traj = np.vstack([ref_traj, traj])
    ref_traj[:, 3] = process_angle(ref_traj[:, 3])
    ref_traj[0, 2] = 0
    ref_traj[-1, 2] = 0

    return ref_traj
