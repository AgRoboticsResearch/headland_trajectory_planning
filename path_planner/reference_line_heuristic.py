from importlib.resources import path
import math
import numpy as np
from orchard_geometry_environment import OrchardGeometryEnvironment
from utils.cubic_spline import calc_spline_course
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from utils.path_utils import get_projection_point, angle_wrap
from car_model import CarModel
from shapely.ops import unary_union


class ReferenceLineHeuristic(object):
    ACCEPT_PATH_DEVIATION = 2
    DRIVE_ROW_OFFSET = 5.0
    LARGE_SEARCH_LENGTH = 1.0

    def __init__(
        self,
        waypoints: list,
        goal_pose: list,
        car_model: CarModel,
        obstacle_polys=[],
        default_search_length=1.5,
    ):
        self.default_search_length = default_search_length
        self.goal_pose = goal_pose
        self.car_model = car_model
        (
            self.guided_path,
            self.guided_lane,
            self.way_points,
            self.segment_lanes,
        ) = self.get_guide_line(waypoints)

        self.search_lengths = self.create_segment_lengths(
            self.segment_lanes, obstacle_polys
        )
        self.segment_lanes_strtree = STRtree(self.segment_lanes)

    def plot_heuristic(self, plt, lane_visual=False):

        plt.plot(self.guided_path[:, 0], self.guided_path[:, 1], "b--")
        if lane_visual:
            for seg_lane in self.segment_lanes:
                plt.fill(*seg_lane.exterior.xy, color="gray", alpha=0.2)
            plt.fill(*self.guided_lane.exterior.xy, color="red", alpha=0.1)
        plt.scatter(self.way_points[:, 0], self.way_points[:, 1], c="red")

    def get_guide_line(self, waypoints):
        segment_lanes = []
        step = 0.1
        # print(way_points)
        way_xs, way_ys = np.array([]), np.array([])
        way_yaws = np.array([])
        for i in range(1, len(waypoints)):
            x_end, x_start = waypoints[i, 0], waypoints[i - 1, 0]
            y_end, y_start = waypoints[i, 1], waypoints[i - 1, 1]
            dist = np.hypot(x_end - x_start, y_end - y_start)
            num = int(dist / step)
            xs = np.linspace(x_start, x_end, num)
            ys = np.linspace(y_start, y_end, num)
            way_xs = np.append(way_xs, xs)
            way_ys = np.append(way_ys, ys)
            segment_lanes.append(
                LineString(waypoints[i - 1 : i + 1]).buffer(6, cap_style=1, join_style=3)
            )
            yaw = math.atan2(y_end - y_start, x_end - x_start)
            yaws = np.ones_like(xs) * yaw
            way_yaws = np.append(way_yaws, yaws)

        delta_xs = np.diff(way_xs)
        delata_ys = np.diff(way_ys)
        delta_ss = np.hypot(delta_xs, delata_ys)
        way_ss = np.zeros_like(way_xs)
        way_ss[1:] = np.cumsum(delta_ss)
        guided_path = np.array([way_xs, way_ys, way_yaws, way_ss]).T
        # rx, ry, ryaw, rk, rs = calc_spline_course(way_xs, way_ys, ds=0.1)
        # guided_path = np.array([rx, ry, ryaw, rk, rs]).T

        guided_lane = unary_union(segment_lanes)
        return guided_path, guided_lane, waypoints, segment_lanes

    def create_segment_lengths(self, segment_lanes, obs_polys):
        search_lengths = np.ones(len(segment_lanes)) * self.default_search_length
        if len(search_lengths) > 4:
            if len(obs_polys) == 0:
                search_lengths[2 : len(search_lengths) - 1] = self.LARGE_SEARCH_LENGTH
            else:
                obs_strtree = STRtree(obs_polys)
                for i in range(2, len(search_lengths) - 1):
                    nearest_obs = obs_strtree.nearest(segment_lanes[i])
                    if not nearest_obs.intersects(segment_lanes[i]):
                        search_lengths[i] = self.LARGE_SEARCH_LENGTH

        return search_lengths

    def aux_polys_are_feasible(self, aux_polys):
        for aux_poly in aux_polys:
            if not self.guided_lane.contains(aux_poly):
                return False

        return True

    def check_path_feasibility(self, car_model: CarModel, path: np.ndarray):
        path_poly, aux_polys = car_model.get_path_poly(path)
        if self.guided_lane.contains(path_poly):
            return True
        # if car_model.with_aux:
        #     if self.guided_lane.contains(path_poly) and self.aux_polys_are_feasible(
        #         aux_polys
        #     ):
        #         return True
        # else:
        #     if self.guided_lane.contains(path_poly):
        #         return True

        return False

    def get_search_length(self, pose):
        # judge which segment of the pose
        point = Point(pose[0], pose[1])
        search_length = self.default_search_length
        for i in range(len(self.segment_lanes)):
            segment_poly = self.segment_lanes[i]
            if segment_poly.contains(point):
                search_length = self.search_lengths[i]

        return search_length

    def calculate_state_cost(self, pose):
        # distance to the line string
        # front_x = pose[0] + self.car_model.WHEEL_BASE * np.cos(pose[2])
        # front_y = pose[1] + self.car_model.WHEEL_BASE * np.sin(pose[2])
        # pose = np.array([front_x, front_y, pose[2]])

        dists = np.hypot(
            self.guided_path[:, 0] - pose[0], self.guided_path[:, 1] - pose[1]
        )
        match_idx = np.argmin(dists)
        match_pose = self.guided_path[match_idx]

        distance_to_path = dists[match_idx] * 100
        yaw_difference = abs(angle_wrap(match_pose[2] - pose[2]))

        if distance_to_path > self.ACCEPT_PATH_DEVIATION:
            distance_to_path = 100

        # cost = dist_cost
        dist_to_goal = self.guided_path[-1, -1] - self.guided_path[match_idx, -1]
        cost = (
            distance_to_path
            + yaw_difference * 0.2
            + dist_to_goal * 5
            # + np.hypot(pose[0] - self.goal_pose[0], pose[1] - self.goal_pose[1])
        )

        return cost
