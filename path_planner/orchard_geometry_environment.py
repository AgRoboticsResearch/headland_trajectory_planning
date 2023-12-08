import math
import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from car_model import CarModel
from shapely.strtree import STRtree
import shapely

# from occupancy_grid_field_map import OccupancyGridFieldMap


class OrchardGeometryEnvironment(object):
    NEAR_SIDE = 1
    FAR_SIDE = -1

    def __init__(
        self,
        map_tree_rows,
        obstacles,
        contour_points=[],
        tree_width=0.2,
        headland_width=7,
        obstacle_dim=0.3,
    ):
        self.map_tree_rows = map_tree_rows
        self.tree_width = tree_width
        self.headland_width = headland_width
        self.tree_polys = self.create_row_polygons(tree_width)
        self.obstacle_polys = self.create_obstacle_polygons(obstacles, obstacle_dim)
        self.field_range_poly = self.create_field_polygon(headland_width, contour_points)
        self.contour_points = contour_points
        self.obs_strtree = STRtree(self.obstacle_polys + self.tree_polys)
        self.obs_poly_list = self.obstacle_polys + self.tree_polys

        # create occupancy grid map instances for the map
        # self.occupancy_gird_map = OccupancyGridFieldMap(
        #     map_tree_rows,
        #     obstacles=obstacles,
        #     contour_points=contour_points,
        #     tree_width=tree_width,
        #     headland_width=headland_width,
        #     obstacle_dim=obstacle_dim,
        #     map_resolution=0.04,
        # )

    def update_tree_width(self, new_tree_width):
        self.tree_polys = self.create_row_polygons(new_tree_width)
        self.obs_poly_list = self.obstacle_polys + self.tree_polys

    def check_side_of_a_point(self, point):
        row_centers = np.mean(self.map_tree_rows[:, :, :], axis=1)
        contour_points = np.array(self.field_range_poly.exterior.coords)

        # line of row centers
        epsilon = np.random.uniform(-0.5, 0.5, size=(len(row_centers),))
        self._center_line_coeff = np.polyfit(
            row_centers[:, 0] + epsilon, row_centers[:, 1], deg=1
        )
        k, b = self._center_line_coeff[0], self._center_line_coeff[1]
        origin_sign = np.sign(0 * k + b - 0)

        row_side_judge = np.sign(point[0] * k + b - point[1])
        near_row_side = self.NEAR_SIDE if origin_sign == row_side_judge else self.FAR_SIDE

        return near_row_side

    def create_headland_countour_lines(self, field_range_poly):
        row_centers = np.mean(self.map_tree_rows[:, :, :], axis=1)
        contour_points = np.array(field_range_poly.exterior.coords)[:-1]

        # line of row centers
        epsilon = np.random.uniform(-0.5, 0.5, size=(len(row_centers),))
        self._center_line_coeff = np.polyfit(
            row_centers[:, 0] + epsilon, row_centers[:, 1], deg=1
        )
        k, b = self._center_line_coeff[0], self._center_line_coeff[1]
        # origin of the map must be on the near side
        self._origin_sign = np.sign(0 * k + b - 0)
        row_side_judges = np.sign(contour_points[:, 0] * k + b - contour_points[:, 1])
        near_row_side = np.where((row_side_judges == self._origin_sign))
        far_row_side = np.where(
            (
                (row_side_judges > self._origin_sign)
                | (row_side_judges < self._origin_sign)
            )
        )
        contour_points_near = contour_points[near_row_side]
        contour_points_far = contour_points[far_row_side]
        # print("contour_points_near: ", len(contour_points_near))
        # print("contour_points_far: ", len(contour_points_far))

        return contour_points_near, contour_points_far

    def get_row_ids_between_start_and_end(self, start_pose, end_pose):
        near_row_xs = self.map_tree_rows[:, 0, 0]
        near_row_ys = self.map_tree_rows[:, 0, 1]
        far_row_xs = self.map_tree_rows[:, 1, 0]
        far_row_ys = self.map_tree_rows[:, 1, 1]
        # which side of the row
        # get row id
        dists = np.abs(start_pose[1] - near_row_ys)
        row_id = np.argmin(dists)
        check_side_ys = (
            np.copy(near_row_ys)
            if abs(start_pose[0] - near_row_xs[row_id])
            < abs(start_pose[0] - far_row_xs[row_id])  # near to near row end
            else np.copy(far_row_ys)
        )

        check_side_xs = (
            np.copy(near_row_xs)
            if abs(start_pose[0] - near_row_xs[row_id])
            < abs(start_pose[0] - far_row_xs[row_id])  # near to near row end
            else np.copy(far_row_xs)
        )

        # get intermediate row idxes
        start_y, end_y = start_pose[1], end_pose[1]
        if start_y > end_y:
            intermediate_row_idx = np.where(
                (check_side_ys > end_y) & (check_side_ys < start_y)
            )[0]
        else:
            intermediate_row_idx = np.where(
                (check_side_ys > start_y) & (check_side_ys < end_y)
            )[0]

        return check_side_xs, check_side_ys, intermediate_row_idx

    def plot_field_geometry(self, plt, color="skyblue", with_range=True):
        for row in self.map_tree_rows:
            plt.plot(row[:, 0], row[:, 1], "*-", c="g")

        # plot geometries
        for tree_poly in self.tree_polys:
            plt.fill(*tree_poly.exterior.xy, color="g", alpha=0.5)

        for obs_poly in self.obstacle_polys:
            plt.fill(*obs_poly.exterior.xy, color="orange", alpha=0.9)

        if with_range:
            plt.fill(*self.field_range_poly.exterior.xy, color=color, alpha=0.5)
        # plt.scatter(self.contour_points_far[:, 0], self.contour_points_far[:, 1], c="r")
        # plt.scatter(
        #     self.contour_points_near[:, 0], self.contour_points_near[:, 1], c="blue"
        # )

    def get_intermediate_contour_points(
        self,
        safety_distance,
        start_point,
        sorted_intermediate_row_xs,
        sorted_intermediate_row_ys,
    ):
        # intermediate contour points based on tree row end
        side_of_start_point = self.check_side_of_a_point(start_point)
        offset = (
            -safety_distance if side_of_start_point == self.NEAR_SIDE else safety_distance
        )
        contour_points = np.vstack(
            (sorted_intermediate_row_xs[:] + offset, sorted_intermediate_row_ys[:])
        ).T

        return contour_points

    def get_topology_waypoints_for_headland_transition(
        self,
        start_pose,
        end_pose,
        drive_row_offset,
    ):
        # get intermediate row ids between start and end position
        (
            check_side_xs,
            check_side_ys,
            intermediate_row_idx,
        ) = self.get_row_ids_between_start_and_end(start_pose, end_pose)

        # tree row end points
        intermediate_row_ys = check_side_ys[intermediate_row_idx]
        intermediate_row_xs = check_side_xs[intermediate_row_idx]

        sorted_idxes = np.argsort(np.abs(intermediate_row_ys - start_pose[1]))
        intermediate_row_ys = intermediate_row_ys[sorted_idxes]
        intermediate_row_xs = intermediate_row_xs[sorted_idxes]

        contour_points = self.get_intermediate_contour_points(
            drive_row_offset, start_pose[:2], intermediate_row_xs, intermediate_row_ys
        )
        way_points = np.vstack(
            (
                start_pose[:2],
                contour_points[1:],
                end_pose[:2],
            )
        )

        return way_points

    def get_topology_waypoints(
        self,
        start_pose,
        end_pose,
        drive_row_offset,
    ):
        # get intermediate row ids between start and end position
        (
            check_side_xs,
            check_side_ys,
            intermediate_row_idx,
        ) = self.get_row_ids_between_start_and_end(start_pose, end_pose)

        # tree row end points
        intermediate_row_ys = check_side_ys[intermediate_row_idx]
        intermediate_row_xs = check_side_xs[intermediate_row_idx]

        sorted_idxes = np.argsort(np.abs(intermediate_row_ys - start_pose[1]))
        intermediate_row_ys = intermediate_row_ys[sorted_idxes]
        intermediate_row_xs = intermediate_row_xs[sorted_idxes]

        contour_points = self.get_intermediate_contour_points(
            drive_row_offset, start_pose[:2], intermediate_row_xs, intermediate_row_ys
        )

        offset = drive_row_offset if np.cos(start_pose[2]) > 0 else -drive_row_offset
        start_ahead_points = np.copy(start_pose[:2])
        start_ahead_points[0] += offset
        end_ahead_points = np.copy(end_pose[:2])
        end_ahead_points[0] += offset
        way_points = np.vstack(
            (
                start_pose[:2],
                contour_points,
                # end_ahead_points,
                end_pose[:2],
            )
        )

        # start_waypoint, goal_waypoint = self.get_start_and_goal_waypoints(
        #     start_pose, end_pose, check_side_xs, check_side_ys
        # )
        # extend_point = np.copy(end_pose)
        # extend_point[0] = goal_waypoint[0] + 5 * np.cos(end_pose[2])
        # way_points = np.vstack((start_waypoint, contour_points, goal_waypoint))
        # way_points = np.vstack(
        #     (start_pose[:2], contour_points, goal_waypoint, extend_point[:2])
        # )

        return way_points

    def get_which_side_of_pose(self, pose):
        k, b = self._center_line_coeff[0], self._center_line_coeff[1]
        sign = np.sign(k * pose[0] + b - pose[1])

        if sign == self._origin_sign:
            return self.NEAR_SIDE
        else:
            return self.FAR_SIDE

    def get_start_and_goal_waypoints(
        self, start_pose, goal_pose, check_side_xs, check_side_ys
    ):
        checkside_drive_row_xs = check_side_xs[:-1] + np.diff(check_side_xs) / 2
        checkside_drive_row_ys = check_side_ys[:-1] + np.diff(check_side_ys) / 2

        start_idx = np.argmin(np.abs(checkside_drive_row_ys - start_pose[1]))
        goal_idx = np.argmin(np.abs(checkside_drive_row_ys - goal_pose[1]))

        start_waypoint = np.array(
            [checkside_drive_row_xs[start_idx], checkside_drive_row_ys[start_idx]]
        )
        goal_waypoint = np.array(
            [checkside_drive_row_xs[goal_idx], checkside_drive_row_ys[goal_idx]]
        )
        return start_waypoint, goal_waypoint

    # create tree polygons
    def create_row_polygons(self, tree_width):
        row_polygons = []
        for row in self.map_tree_rows:
            # no extra on cap
            row_polygon = LineString(row).buffer(
                tree_width / 2, cap_style=2, join_style=1
            )
            row_polygons.append(row_polygon)

        return row_polygons

    def get_map_exterior_pts(self, headland_width):
        row_width = np.mean(np.diff(self.map_tree_rows[:, 0, 1]))
        near_angle = self.get_headland_angle(self.NEAR_SIDE)
        far_angle = self.get_headland_angle(self.FAR_SIDE)
        delta_x_near = abs(headland_width / math.sin(near_angle))
        delta_x_far = abs(headland_width / math.sin(far_angle))
        # print("delta_x_far: ", delta_x_far)
        # print("delta_x_near: ", delta_x_near)
        # print("headland_width: ", headland_width)

        # near side
        near_end_points = []
        for tree_row in self.map_tree_rows:
            near_end_points.append(tree_row[0])
        near_end_points = np.array(near_end_points)
        near_end_points[:, 0] -= delta_x_near
        near_end_points_up_idx = np.argmax(near_end_points[:, 1])
        near_end_points[near_end_points_up_idx, 1] += row_width
        if np.abs(np.sin(near_angle)) < 1e-5:
            delta_x = 0
        else:
            delta_x = row_width / np.tan(near_angle)
        near_end_points[near_end_points_up_idx, 0] += delta_x
        near_end_points_low_idx = np.argmin(near_end_points[:, 1])
        near_end_points[near_end_points_low_idx, 1] -= row_width
        near_end_points[near_end_points_low_idx, 0] -= delta_x
        # far side
        far_end_points = []
        for tree_row in self.map_tree_rows:
            far_end_points.append(tree_row[1])
        far_end_points.reverse()
        far_end_points = np.array(far_end_points)
        far_end_points[:, 0] += delta_x_far
        far_end_points_up_idx = np.argmax(far_end_points[:, 1])
        if np.abs(np.sin(far_angle)) < 1e-5:
            delta_x = 0
        else:
            delta_x = row_width / np.tan(far_angle)
        far_end_points[far_end_points_up_idx, 1] += row_width
        far_end_points[far_end_points_up_idx, 0] += delta_x
        far_end_points_low_idx = np.argmin(far_end_points[:, 1])
        far_end_points[far_end_points_low_idx, 1] -= row_width
        far_end_points[far_end_points_low_idx, 0] -= delta_x

        exterior_points = np.concatenate((near_end_points, far_end_points))

        return exterior_points

    def create_field_polygon(self, headland_width, contour_points):
        if len(contour_points) == 0:
            exterior_points = self.get_map_exterior_pts(headland_width)
            poly = Polygon(exterior_points)
        else:
            poly = Polygon(contour_points)

        return poly

    def create_obstacle_polygons(self, obstacles, obstacle_dim):
        obs_polys = []
        if len(obstacles) > 0:
            for obs in obstacles:
                x, y = obs[0], obs[1]
                obs_poly = Point(x, y).buffer(obstacle_dim, cap_style="square")
                obs_polys.append(obs_poly)

        return obs_polys

    def tree_row_collision_check(
        self,
        path_poly,
    ):
        # ASSUMPTION: x is along row extension direction, y is along cross row
        for row_poly in self.tree_polys:
            if path_poly.intersects(row_poly):
                return True

        return False

    def obstacles_collision_check(self, path_poly: Polygon):
        if len(self.obstacle_polys) > 0:
            for obs_poly in self.obstacle_polys:
                if path_poly.intersects(obs_poly):
                    return True

        return False

    def out_of_field_boundary_check(self, path_poly: Polygon):
        if not self.field_range_poly.contains(path_poly):
            return True

        return False

    def get_distance_to_nearest_poly(self, car_model: CarModel, path):
        path_poly, _ = car_model.get_path_poly(path)

        nearest_poly = self.obs_strtree.nearest(path_poly)
        # nearest_poly = self.obs_poly_list[nearest_poly_idx]
        if nearest_poly.intersects(path_poly):
            return 0

        dist = nearest_poly.distance(path_poly)
        # TODO: check the distance to the field range

        return dist

    def get_min_distance_to_boundary(self, car_model: CarModel, path, with_aux=True):
        path_poly, aux_path_polys = car_model.get_path_poly(path)
        exterior_coords = list(path_poly.exterior.coords)
        if with_aux and len(aux_path_polys) > 0:
            for aux_path_poly in aux_path_polys: 
                exterior_coords += aux_path_poly.exterior.coords
        field_poly_exterior = self.field_range_poly.exterior
        min_distance = np.inf

        for coords in exterior_coords:
            point = Point(coords)
            distance = point.distance(field_poly_exterior)
            # point outside the field will have negative distance
            # if the path is outside, it will return the maximum negative distance
            if not self.field_range_poly.contains(point):
                distance *= -1
            if distance < min_distance:
                min_distance = distance
                               
        return min_distance
        
    def get_nearest_poly_of_a_geometry(self, geometry: Polygon):
        if int(shapely.__version__[0]) == 2:
            nearest_idxs = self.obs_strtree.query_nearest(geometry)
            nearest_poly = self.obs_strtree.geometries[nearest_idxs[0]]
        else:
            nearest_poly = self.obs_strtree.nearest(geometry)

        return nearest_poly

    def check_path_feasibility(
        self, car_model: CarModel, path: np.ndarray, boundary_check=True, aux_check=False
    ):
        path_poly, aux_path_polys = car_model.get_path_poly(path)

        nearest_poly = self.get_nearest_poly_of_a_geometry(path_poly)
        if nearest_poly.intersects(path_poly):
            # print("obstacles intersected with main body")
            return False

        # out of field check
        if self.out_of_field_boundary_check(path_poly) and boundary_check:
            # print("out of field")
            return False

        # aux part collision
        if len(aux_path_polys) > 0 and aux_check:
            for aux_path_poly in aux_path_polys:
                if type(aux_path_poly) == MultiPolygon:
                    for poly in aux_path_poly.geoms:
                        nearest_poly = self.get_nearest_poly_of_a_geometry(poly)
                        # nearest_poly = self.obs_poly_list[nearest_poly_idx]
                        if poly.intersects(nearest_poly):
                            # print("obstacles intersected with aux part")
                            return False
                if type(aux_path_poly) == Polygon:
                    nearest_poly = self.get_nearest_poly_of_a_geometry(aux_path_poly)
                    if aux_path_poly.intersects(nearest_poly):
                        # print("obstacles intersected with aux part")
                        return False

                # DO NOT check aux part for boundary
                if self.out_of_field_boundary_check(aux_path_poly) and boundary_check:
                    return False

        return True

    # def check_path_feasibility_with_grid_map(self, car: CarModel, path: np.ndarray):
    #     self.occupancy_gird_map.check_path_feasibility_in_local_map(car, path)

    def get_headland_angle(self, side):
        side_idx = 0 if side == self.NEAR_SIDE else 1
        row_end_points_xs = self.map_tree_rows[:, side_idx, 0]
        if np.std(row_end_points_xs) < 0.01:
            return np.pi / 2
        else:
            row_end_points_ys = self.map_tree_rows[:, side_idx, 1]
            line_coeff = np.polyfit(row_end_points_xs, row_end_points_ys, deg=1)
            k, _ = line_coeff[0], line_coeff[1]
            return math.atan(k)
