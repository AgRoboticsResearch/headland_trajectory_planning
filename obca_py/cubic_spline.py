import scipy.interpolate as scipy_interpolate
from scipy.interpolate import CubicSpline
import math
import numpy as np
import bisect

# this is equal to matlab function: spline
def spline_python(xs, ys, ts, deg=3):
    if len(xs) < 3:
        tck = scipy_interpolate.splrep(xs, ys, k=len(xs) - 1)
    else:
        tck = scipy_interpolate.splrep(xs, ys, k=deg)

    y_preds = scipy_interpolate.splev(ts, tck, der=0)

    return y_preds


class Spline2D:
    """
    2D Cubic Spline class

    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline(self.s, x)
        self.sy = CubicSpline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))

        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx(s)
        y = self.sy(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = np.asarray(self.sx(s, 1))
        dy = np.asarray(self.sy(s, 1))
        ddx = np.asarray(self.sx(s, 2))
        ddy = np.asarray(self.sy(s, 2))

        k = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** (3.0 / 2.0))
        # print("k: ", k)
        return k

    def calc_curvature_prime(self, s):
        """
        calc curvature
        """
        dx = self.sx(s, 1)
        dy = self.sy(s, 1)
        ddx = self.sx(s, 2)
        ddy = self.sy(s, 2)
        dddx = self.sx(s, 3)
        dddy = self.sy(s, 3)

        p1 = dx**2 + dy**2
        p2 = dx * dddy - dy * dddx
        p3 = dx * ddy - ddx * dy
        p4 = dx * ddx + dy * ddy
        bottom = (dx**2 + dy**2) ** (5 / 2)
        k_prime = (p1 * p2 - 3 * p3 * p4) / bottom

        return k_prime

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx(s, 1)
        dy = self.sy(s, 1)
        yaw = np.arctan2(dy, dx)

        return yaw


def calc_spline_course(x, y, ds=0.1):
    # remove the singular points points
    diff_xs = np.diff(x)
    diff_ys = np.diff(y)
    singular_idx = np.where((diff_xs == 0) & (diff_ys == 0))
    if len(singular_idx) > 0:
        x = np.delete(x, singular_idx, axis=0)
        y = np.delete(y, singular_idx, axis=0)

    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1] + ds, ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s
