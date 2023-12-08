import math
import numpy as np

# will be imported from navigation_utils
def calculate_path_length(xs, ys):
    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.hypot(dx, dy)
    ss = np.cumsum(ds)
    path_length = ss[-1]

    return path_length


def get_projection_point(x_m, y_m, yaw_m, k_m, x, y):

    # get projection point on the curve
    d_vector = np.array([x - x_m, y - y_m])
    tau_vector = np.array([math.cos(yaw_m), math.sin(yaw_m)])
    p_proj = np.array([x_m, y_m]) + (d_vector.dot(tau_vector)) * tau_vector
    yaw_proj = yaw_m + k_m * (d_vector.dot(tau_vector))

    return p_proj, yaw_proj


def angle_wrap(angles):
    # output angle between -pi to pi
    wrap_angle = (angles + math.pi) % (2 * math.pi) - math.pi
    return wrap_angle
