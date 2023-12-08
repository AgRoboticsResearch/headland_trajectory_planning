from shapely.geometry import Polygon, Point
import numpy as np
import math
from shapely.ops import unary_union
import matplotlib.path as mpltPath
from utils.path_utils import angle_wrap


class CarModel:
    def __init__(
        self,
        max_steer=0.55,  # 0.60
        wheel_base=1.9,
        axle_to_front=2.85,
        axle_to_back=0.5,  # 0.96
        width=1.48,  # 1.55
        head_out=0.542,
        head_side=0.44,
        body_vertices=[],
        aux_poly_features=[],
        with_aux=False,
    ):
        # kinematic of car
        self.MAX_STEER = max_steer
        self.WHEEL_BASE = wheel_base
        self.aux_polys = []

        # collision check of the car
        self.AXLE_TO_FRONT = axle_to_front
        self.AXLE_TO_BACK = axle_to_back
        self.WIDTH = width
        self.HEAD_OUT = head_out
        self.HEAD_SIDE = head_side
        self.curvature = math.tan(self.MAX_STEER) / self.WHEEL_BASE
        self.with_aux = with_aux
        self.body_vertices = body_vertices
        self.get_car_poly(aux_poly_features)

    def get_path_poly(self, path: np.ndarray, skip=1):
        car_polys = []
        aux_path_polys = []
        for pose in path[0 : len(path) : skip]:
            x, y, yaw = pose[0], pose[1], pose[2]
            car_poly_points = np.array(self.car_poly.exterior.xy)

            rotationZ = np.array(
                [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
            )
            car_poly_points = np.dot(rotationZ, car_poly_points)
            car_poly_points += np.array([[x], [y]])
            car_polys.append(Polygon(car_poly_points.T))

        path_poly = unary_union(car_polys)

        if len(self.aux_polys) > 0:
            for aux_poly in self.aux_polys:
                aux_polys = []
                for pose in path[0 : len(path) : 2]:
                    x, y, yaw = pose[0], pose[1], pose[2]
                    rotationZ = np.array(
                        [
                            [math.cos(yaw), -math.sin(yaw)],
                            [math.sin(yaw), math.cos(yaw)],
                        ]
                    )
                    aux_poly_points = np.array(aux_poly.exterior.xy)
                    aux_poly_points = np.dot(rotationZ, aux_poly_points)
                    aux_poly_points += np.array([[x], [y]])
                    aux_polys.append(Polygon(aux_poly_points.T))
                aux_path_poly = unary_union(aux_polys)
                aux_path_polys.append(aux_path_poly)

        return path_poly, aux_path_polys

    def get_car_poly(self, aux_poly_features):
        # main_body = np.array(
        #     [
        #         [
        #             -self.AXLE_TO_BACK,
        #             -self.AXLE_TO_BACK,
        #             self.AXLE_TO_FRONT - self.HEAD_OUT,
        #             self.AXLE_TO_FRONT - self.HEAD_OUT,
        #             self.AXLE_TO_FRONT,
        #             self.AXLE_TO_FRONT,
        #             self.AXLE_TO_FRONT - self.HEAD_OUT,
        #             self.AXLE_TO_FRONT - self.HEAD_OUT,
        #             -self.AXLE_TO_BACK,
        #         ],
        #         [
        #             self.WIDTH / 2,
        #             -self.WIDTH / 2,
        #             -self.WIDTH / 2,
        #             -self.WIDTH / 2 + self.HEAD_SIDE,
        #             -self.WIDTH / 2 + self.HEAD_SIDE,
        #             self.WIDTH / 2 - self.HEAD_SIDE,
        #             self.WIDTH / 2 - self.HEAD_SIDE,
        #             self.WIDTH / 2,
        #             self.WIDTH / 2,
        #         ],
        #     ]
        # )
        main_body = np.array(
            [
                [
                    -self.AXLE_TO_BACK,
                    -self.AXLE_TO_BACK,
                    self.AXLE_TO_FRONT,
                    self.AXLE_TO_FRONT,
                    -self.AXLE_TO_BACK,
                ],
                [
                    self.WIDTH / 2,
                    -self.WIDTH / 2,
                    -self.WIDTH / 2,
                    self.WIDTH / 2,
                    self.WIDTH / 2,
                ],
            ]
        )
        self.car_poly = Polygon(main_body.T)
        self.plt_car_poly = mpltPath.Path(main_body.T)

        if self.with_aux:
            self.aux_polys = []
            self.plt_aux_polys = []
            aux_polys = self.get_aux_shapely_polys(aux_poly_features)
            # attached_parts = []
            splitted_parts = []
            splitted_parts_plt = []
            for aux_part in aux_polys:
                # if aux_part.intersects(self.car_poly):
                #     # attached_parts.append(aux_part)
                #     self.car_poly = unary_union([self.car_poly, aux_part])
                # else:
                splitted_parts.append(aux_part)
                splitted_parts_plt.append(mpltPath.Path(aux_part.exterior.coords))

            # self.car_poly = unary_union([self.car_poly] + attached_parts)
            self.plt_car_poly = mpltPath.Path(self.car_poly.exterior.coords)
            if len(splitted_parts) > 0:
                self.aux_polys = splitted_parts
                self.plt_aux_polys = splitted_parts_plt

        return

    def get_aux_shapely_polys(self, aux_polys):
        """
        aux_polys: [feature]
        feature[0]: [x, y] of left top vertice
        feature[1]: height of the rectangle
        feature[2]: width of the rectangle
        """
        shapely_polys = []
        for feature in aux_polys:
            p1 = np.array([feature[0][0], feature[0][1]])
            p2 = np.array([feature[0][0] + feature[2], feature[0][1]])
            p3 = np.array([feature[0][0] + feature[2], feature[0][1] - feature[1]])
            p4 = np.array([feature[0][0], feature[0][1] - feature[1]])
            shapely_poly = Polygon(np.array([p1, p2, p3, p4]))
            shapely_polys.append(shapely_poly)

        return shapely_polys

    def draw_car(self, plt, x, y, yaw, color="black", alpha=0.1):
        car_poly, aux_polys = self.get_car_poly_in_odom(x, y, yaw)
        car = np.asarray(car_poly.exterior.xy)
        plt.plot(car[0, :], car[1, :], color, alpha=alpha)
        plt.fill(*car, color="orange", alpha=alpha)

        if len(self.aux_polys) > 0:
            for aux_poly in aux_polys:
                aux = np.array(aux_poly.exterior.xy)
                plt.plot(aux[0, :], aux[1, :], color, alpha=alpha)
                plt.fill(*aux, color="orange", alpha=alpha)

        return car_poly, aux_polys

    def get_car_poly_in_odom(self, odom_x, odom_y, odom_yaw):
        rotationZ = np.array(
            [
                [math.cos(odom_yaw), -math.sin(odom_yaw)],
                [math.sin(odom_yaw), math.cos(odom_yaw)],
            ]
        )
        car = np.array(self.car_poly.exterior.xy)

        car = np.dot(rotationZ, car)
        car += np.array([[odom_x], [odom_y]])
        car_poly_odom = Polygon(car.T)

        aux_polys_odom = []
        if len(self.aux_polys) > 0:
            for aux_poly in self.aux_polys:
                aux = np.array(aux_poly.exterior.xy)
                aux = np.dot(rotationZ, aux)
                aux += np.array([[odom_x], [odom_y]])
                aux_poly_odom = Polygon(aux.T)
                aux_polys_odom.append(aux_poly_odom)

        return car_poly_odom, aux_polys_odom

    def calculate_motion_path(self, init_pose, motion_command, delta_yaw, step):
        steer_angle = motion_command[0]
        speed_direction = motion_command[1]
        search_length = delta_yaw / self.curvature
        num_steps = round(search_length / step)
        yaw_step = speed_direction * step / self.WHEEL_BASE * math.tan(steer_angle)

        init_yaw = angle_wrap(init_pose[-1] + yaw_step)
        init_x = init_pose[0]
        init_y = init_pose[1]

        yaws = np.linspace(
            init_yaw, init_yaw + yaw_step * (num_steps + 1), num_steps + 1
        )
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
            curvature = math.tan(motion_command[0]) / self.WHEEL_BASE

        ks = np.ones((len(path), 1)) * curvature
        dirs = np.ones((len(path), 1)) * motion_command[1]
        path = np.hstack((path, ks, dirs))

        return path

    def calculate_motion_path_new(
        self, init_pose, motion_dir, steer_dir, turning_radius, delta_yaw, step_size=0.1
    ):
        turning_radius = max(1.0 / self.curvature, turning_radius)
        steer_angle = math.atan(self.WHEEL_BASE / turning_radius) * steer_dir
        arc_length = abs(delta_yaw * turning_radius)
        num_steps = int(arc_length / step_size)
        actual_step_size = arc_length / num_steps
        yaw_step = (
            motion_dir * actual_step_size / self.WHEEL_BASE * math.tan(steer_angle)
        )

        init_x = init_pose[0]
        init_y = init_pose[1]
        init_yaw = angle_wrap(init_pose[-1])
        yaws = np.linspace(init_yaw, init_yaw + yaw_step * num_steps, num_steps + 1)
        yaws = angle_wrap(yaws)

        xs = init_x + turning_radius * (np.sin(yaws) - np.sin(init_yaw)) * steer_dir
        ys = init_y - turning_radius * (np.cos(yaws) - np.cos(init_yaw)) * steer_dir

        path = np.vstack([xs, ys, yaws]).T
        path = np.vstack([init_pose, path])

        # add curvature and dirs
        curvature = 0
        if abs(steer_angle) > 0.00001:
            curvature = math.tan(steer_angle) / self.WHEEL_BASE

        ks = np.ones((len(path), 1)) * curvature
        dirs = np.ones((len(path), 1)) * motion_dir
        path = np.hstack((path, ks, dirs))

        return path

    def get_turn_radius(self, max_steer_angle=None):
        if max_steer_angle == None:
            return 1 / self.curvature
        else:
            return self.WHEEL_BASE / math.tan(max_steer_angle)

    def get_steer_angle(self, curvature):
        steer_angle = min(math.atan(self.WHEEL_BASE * curvature), self.MAX_STEER)
        return steer_angle
