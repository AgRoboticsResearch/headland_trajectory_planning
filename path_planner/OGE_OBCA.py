import math
from orchard_geometry_environment import OrchardGeometryEnvironment
import numpy as np
from pypoman import plot_polygon
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from scipy.spatial import ConvexHull
from rdp import rdp


def shortest_distance(
    x1: np.ndarray, y1: np.ndarray, a: float, b: float, c: float
) -> float:
    return np.abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))


def point_along_centerline(A, B, d):
    M = (A + B) / 2.0
    L = np.linalg.norm(B - A)
    N = np.array([(B[1] - A[1]) / L, -(B[0] - A[0]) / L])
    offset_vector = N * d
    return M + offset_vector


def point_side_of_line(A, B, C):
    AB = (B[0] - A[0], B[1] - A[1])
    AC = (C[0] - A[0], C[1] - A[1])
    cross_product = AB[0] * AC[1] - AB[1] * AC[0]
    return np.sign(cross_product)


def points_along_rectangles(A, B, d):
    """
    D --- C
    |     |  } -> d
    A --- B
    """
    extra_point = point_along_centerline(A, B, d)
    C = extra_point + (A - B) / 2
    D = extra_point - (A - B) / 2
    return C, D


# inhereitance from orchard geometry environment
class orchard_environment_OBCA(OrchardGeometryEnvironment):
    MIN_ROW_WIDTH = 0.5
    SAFETY_BOUND = 0.2

    def __init__(
        self,
        map_tree_rows,
        obstacles,
        contour_points=[],
        tree_width=0.5,
        headland_width=7,
        obstacle_dim=0.3,
    ):
        # create the orchard geometry environment
        super().__init__(
            map_tree_rows,
            obstacles,
            contour_points=contour_points,
            tree_width=tree_width,
            headland_width=headland_width,
            obstacle_dim=obstacle_dim,
        )
        # (
        #     self.contour_points_near,
        #     self.contour_points_far,
        # ) = self.create_headland_countour_lines()
        self.row_width = np.abs(np.mean(np.diff(self.map_tree_rows[:, 0, 1])))

    # # This is a conservative way
    # def cover_side_points_with_rectangle(self, contour_points, side):
    #     def shortest_distance(
    #         x1: np.ndarray, y1: np.ndarray, a: float, b: float, c: float
    #     ):
    #         d = np.abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))

    #         return d

    #     std_x = np.std(contour_points[:, 0])
    #     up_index = np.argmax(contour_points[:, 1])
    #     down_index = np.argmin(contour_points[:, 1])
    #     extra_point_up = np.copy(contour_points[up_index])
    #     extra_point_down = np.copy(contour_points[down_index])
    #     # straight line: row perpendicular to the boundary
    #     if std_x < 0.1:
    #         if side == self.NEAR_SIDE:
    #             extra_point_up[0] -= 2
    #             extra_point_down[0] -= 2
    #         else:
    #             extra_point_up[0] += 2
    #             extra_point_down[0] += 2
    #         obstacle_contour = np.vstack(
    #             [
    #                 extra_point_up,
    #                 contour_points[up_index],
    #                 contour_points[down_index],
    #                 extra_point_down,
    #             ]
    #         )
    #         obstacle_contour[:2, 1] += 2
    #         obstacle_contour[2:, 1] -= 2
    #     # straight line
    #     else:
    #         line_coeff = np.polyfit(contour_points[:, 1], contour_points[:, 0], deg=1)
    #         # side_index = 0 if side == self.NEAR_SIDE else 1
    #         # side_points = self.map_tree_rows[:, side_index, :]
    #         # # print("side points: ", side_points)
    #         # line_coeff = np.polyfit(side_points[:, 0], side_points[:, 1], deg=1)
    #         k, b = line_coeff[0], line_coeff[1]

    #         if side == self.NEAR_SIDE:
    #             right_points_idxs = np.where(
    #                 contour_points[:, 0] - k * contour_points[:, 1] - b >= 0
    #             )[0]
    #             right_points = contour_points[right_points_idxs]
    #             # get the points most away from line and on the right side of the line
    #             right_point_distances_to_line = shortest_distance(
    #                 right_points[:, 0], right_points[:, 1], 1, -k, -b
    #             )
    #             rightest_indx = right_points_idxs[
    #                 np.argmax(right_point_distances_to_line)
    #             ]
    #             rightest_distance = np.max(right_point_distances_to_line)
    #             # TODO: if the distance is over a threshold, use triangle to represent the boundary
    #             rightest_point = contour_points[rightest_indx, :]
    #             b_max = rightest_point[0] - k * rightest_point[1]
    #             max_y = np.max(contour_points[:, 1])
    #             min_y = np.min(contour_points[:, 1])
    #             extra_point_up_x = max_y * k + b_max
    #             extra_point_down_x = min_y * k + b_max
    #             obstacle_contour = np.array(
    #                 [
    #                     [extra_point_up_x - 2, max_y],
    #                     [extra_point_up_x, max_y],
    #                     [extra_point_down_x, min_y],
    #                     [extra_point_down_x - 2, min_y],
    #                 ]
    #             )
    #         else:
    #             left_points_idxs = np.where(
    #                 contour_points[:, 0] - k * contour_points[:, 1] - b <= 0
    #             )[0]
    #             left_points = contour_points[left_points_idxs]
    #             # get the points most away from line and on the right side of the line
    #             left_point_distances_to_line = shortest_distance(
    #                 left_points[:, 0], left_points[:, 1], 1, -k, -b
    #             )

    #             leftest_indx = left_points_idxs[np.argmax(left_point_distances_to_line)]
    #             leftest_distance = np.max(left_point_distances_to_line)
    #             # TODO: if the distance is over a threshold, use triangle to represent the boundary
    #             leftest_point = contour_points[leftest_indx, :]
    #             b_max = leftest_point[0] - k * leftest_point[1]
    #             max_y = np.max(contour_points[:, 1])
    #             min_y = np.min(contour_points[:, 1])
    #             extra_point_up_x = max_y * k + b_max
    #             extra_point_down_x = min_y * k + b_max
    #             obstacle_contour = np.array(
    #                 [
    #                     [extra_point_up_x + 2, max_y],
    #                     [extra_point_up_x, max_y],
    #                     [extra_point_down_x, min_y],
    #                     [extra_point_down_x + 2, min_y],
    #                 ]
    #             )
    #     return obstacle_contour

    def cover_side_points(self, contour_points, side, width=2):
        std_x = np.std(contour_points[:, 0])
        shift = -width if side == self.NEAR_SIDE else width
        # if the fitted line are perpendicular to x-axis or only two points
        if std_x < 1e-3 or len(contour_points) == 2:
            up_index = np.argmax(contour_points[:, 1])
            down_index = np.argmin(contour_points[:, 1])
            extra_point_up = np.copy(contour_points[up_index])
            extra_point_down = np.copy(contour_points[down_index])
            extra_point_up[0] += shift
            extra_point_down[0] += shift
            obstacle_contour = [
                np.array(
                    [
                        extra_point_up,
                        contour_points[up_index],
                        contour_points[down_index],
                        extra_point_down,
                    ]
                )
            ]

        else:  # otherwise
            k, b = np.polyfit(contour_points[:, 1], contour_points[:, 0], deg=1)
            if side == self.NEAR_SIDE:
                side_condition = lambda x, y: x - k * y - b >= 0
            else:
                side_condition = lambda x, y: x - k * y - b <= 0
            side_points_idxs = np.where(
                np.apply_along_axis(
                    side_condition, 0, contour_points[:, 0], contour_points[:, 1]
                )
            )[0]
            side_points = contour_points[side_points_idxs]
            side_point_distances_to_line = shortest_distance(
                side_points[:, 0], side_points[:, 1], 1, -k, -b
            )
            # if the averaged distance is small enough, use a single quadrilateral
            if side_point_distances_to_line.mean() < 0.1:
                furthest_indx = side_points_idxs[np.argmax(side_point_distances_to_line)]
                furthest_point = contour_points[furthest_indx, :]
                b_max = furthest_point[0] - k * furthest_point[1]
                max_y = np.max(contour_points[:, 1])
                min_y = np.min(contour_points[:, 1])
                extra_point_up_x = max_y * k + b_max
                extra_point_down_x = min_y * k + b_max
                obstacle_contour = [
                    np.array(
                        [
                            [extra_point_up_x + shift, max_y],
                            [extra_point_up_x, max_y],
                            [extra_point_down_x, min_y],
                            [extra_point_down_x + shift, min_y],
                        ]
                    )
                ]
            else:  # otherwise use multiple triangles
                obstacle_contour = []
                dist_signed = -width if side == self.NEAR_SIDE else width
                y_dir = np.sign(contour_points[1][1] - contour_points[0][1])
                dist_signed *= y_dir
                for i in range(len(contour_points) - 1):
                    # 1) use triangles (not recommended for collision)
                    # point = point_along_centerline(
                    #     contour_points[i, :], contour_points[i + 1, :], dist_signed
                    # )
                    # obstacle_contour.append(
                    #     np.array(
                    #         [
                    #             contour_points[i, :],
                    #             contour_points[i + 1, :],
                    #             point,
                    #         ]
                    #     )
                    # )

                    # 2) use rectangles
                    point1, point2 = points_along_rectangles(
                        contour_points[i, :], contour_points[i + 1, :], dist_signed
                    )
                    obstacle_contour.append(
                        np.array(
                            [
                                contour_points[i, :],
                                contour_points[i + 1, :],
                                point1,
                                point2,
                            ]
                        )
                    )

        return obstacle_contour

    def create_inner_polygon(self, near_dist, far_dist):
        near_end_points = []
        for tree_row in self.map_tree_rows:
            near_end_points.append(tree_row[0])
        near_end_points = np.array(near_end_points)
        near_end_points[:, 0] -= near_dist
        # far side
        far_end_points = []
        for tree_row in self.map_tree_rows:
            far_end_points.append(tree_row[1])
        far_end_points.reverse()
        far_end_points = np.array(far_end_points)
        far_end_points[:, 0] += far_dist

        exterior_points = np.concatenate((near_end_points, far_end_points))
        inner_polygon = Polygon(exterior_points)

        return inner_polygon

    def get_inner_polygon_in_field_bounds(self):
        # extend near side
        min_dist = 4
        max_dist = 9
        for dist in np.arange(min_dist, max_dist, 0.1):
            line_points = np.copy(self.map_tree_rows[1:-1, 0, :])
            line_points[:, 0] -= dist
            line_seg = LineString(line_points)
            if not self.field_range_poly.contains(line_seg):
                near_dist = dist
                break
        for dist in np.arange(min_dist, max_dist, 0.2):
            line_points = np.copy(self.map_tree_rows[1:-1, 1, :])
            line_points[:, 0] += dist
            line_seg = LineString(line_points)
            if not self.field_range_poly.contains(line_seg):
                far_dist = dist
                break
        inner_polygon = self.create_inner_polygon(near_dist, far_dist)

        return inner_polygon

    # the field is surrounded with four polygons
    def create_boundary_polygons(self):
        # if len(self.contour_points) > 0:
        #     (field_poly) = self.get_inner_polygon_in_field_bounds()
        #     contour_points_near, contour_points_far = self.create_headland_countour_lines(
        #         field_poly
        #     )
        # else:
        #     contour_points_near, contour_points_far = self.create_headland_countour_lines(
        #         self.field_range_poly
        #     )

        contour_points_near, contour_points_far = self.create_headland_countour_lines(
            self.field_range_poly
        )

        # reduce number of points
        epsilon = 0.15
        contour_points_near_reduced = rdp(contour_points_near, epsilon)
        contour_points_far_reduced = rdp(contour_points_far, epsilon)

        obstacle_near = self.cover_side_points(
            contour_points_near_reduced, self.NEAR_SIDE
        )
        obstacle_far = self.cover_side_points(contour_points_far_reduced, self.FAR_SIDE)

        # bound on the last row
        up_bound_indices = np.argmax(self.map_tree_rows[:, 0, 1])
        up_bound_point_near = np.copy(self.map_tree_rows[up_bound_indices, 0, :])
        up_bound_point_far = np.copy(self.map_tree_rows[up_bound_indices, 1, :])
        up_bound_point_near[1] += self.row_width
        up_bound_point_far[1] += self.row_width
        up_bound_point_near[0] -= 8
        up_bound_point_far[0] += 8
        up_bound_point_extend_near = np.copy(up_bound_point_near)
        up_bound_point_extend_far = np.copy(up_bound_point_far)
        up_bound_point_extend_near[1] += 1
        up_bound_point_extend_far[1] += 1

        obstacle_up = np.vstack(
            [
                up_bound_point_near,
                up_bound_point_extend_near,
                up_bound_point_extend_far,
                up_bound_point_far,
            ]
        )

        low_bound_indices = np.argmin(self.map_tree_rows[:, 0, 1])
        low_bound_point_near = np.copy(self.map_tree_rows[low_bound_indices, 0, :])
        low_bound_point_far = np.copy(self.map_tree_rows[low_bound_indices, 1, :])
        low_bound_point_near[1] -= self.row_width
        low_bound_point_far[1] -= self.row_width
        low_bound_point_near[0] -= 8
        low_bound_point_far[0] += 8
        low_bound_point_extend_near = np.copy(low_bound_point_near)
        low_bound_point_extend_far = np.copy(low_bound_point_far)
        low_bound_point_extend_near[1] -= 1
        low_bound_point_extend_far[1] -= 1
        obstacle_low = np.vstack(
            [
                low_bound_point_near,
                low_bound_point_extend_near,
                low_bound_point_extend_far,
                low_bound_point_far,
            ]
        )

        return obstacle_near, obstacle_far, [obstacle_low], [obstacle_up]

    def polygon_to_convex_sets(self, coords):
        # Convert the input coordinates to a numpy array of points
        points = np.array(coords)

        # Use the ConvexHull function to compute the convex hull of the points
        hull = ConvexHull(points)

        # Extract the vertices of the convex hull
        convex_hull = np.array([points[vertex] for vertex in hull.vertices])

        # If the convex hull is a line segment, return it as a list of two points
        if len(convex_hull) == 2:
            return [convex_hull]

        # Otherwise, create a Shapely Polygon object from the convex hull
        poly = Polygon(convex_hull)

        # If the polygon is already convex, return it as a list
        if len(poly.interiors) == 0:
            return [convex_hull]
        else:
            # Extract the exterior ring and any interior rings (holes)
            exterior = np.array(list(poly.exterior.coords))
            interiors = [np.array(list(interior.coords)) for interior in poly.interiors]

            # Compute the convex hull of the exterior ring and any interior rings
            convex_exterior = Polygon(exterior).convex_hull
            convex_interiors = [Polygon(interior).convex_hull for interior in interiors]

            # Convert each convex hull into a 2D numpy array and return as a list of arrays
            convex_sets = [np.array(list(convex_exterior.exterior.coords))]
            for interior in convex_interiors:
                convex_sets.append(np.array(list(interior.exterior.coords)))

            return convex_sets

    def get_tree_row_obstacles(self, start_pose, end_pose):
        larger_y = max(start_pose[1], end_pose[1])
        smaller_y = min(start_pose[1], end_pose[1])
        intermediate_rows_idxs = np.where(
            (self.map_tree_rows[:, 0, 1] > smaller_y)
            & (self.map_tree_rows[:, 0, 1] < larger_y)
        )[0]
        intermediate_rows_idxs = np.sort(intermediate_rows_idxs)
        intermediate_row_low_idx = intermediate_rows_idxs[0]
        intermediate_row_up_idx = intermediate_rows_idxs[-1]

        intermediate_low_vertices = self.map_tree_rows[intermediate_row_low_idx, :, :]
        low_vertices = np.copy(intermediate_low_vertices)
        low_vertices[:, 1] -= self.tree_width / 2.0

        intermediate_up_vertices = self.map_tree_rows[intermediate_row_up_idx, :, :]
        up_vertice = np.copy(intermediate_up_vertices)
        up_vertice[:, 1] += self.tree_width / 2.0

        intermediate_row_obstacles = np.concatenate(
            [
                low_vertices,
                self.map_tree_rows[intermediate_rows_idxs, 1, :],
                up_vertice,
                self.map_tree_rows[intermediate_rows_idxs[::-1], 0, :],
            ]
        )

        intermediate_row_obstacles = self.polygon_to_convex_sets(
            intermediate_row_obstacles
        )

        relevant_rows_low_idx = max(intermediate_row_low_idx - 1, 0)
        relevant_rows_up_idx = min(
            intermediate_row_up_idx + 1, len(self.map_tree_rows) - 1
        )

        row_obstacles = []
        row_idxs = [relevant_rows_up_idx, relevant_rows_low_idx]
        for row in self.map_tree_rows[row_idxs]:
            # no extra on cap
            near_row_vertice = np.copy(row[0])
            far_row_vertice = np.copy(row[1])

            vertices1 = np.copy(near_row_vertice)
            vertices1[0] -= self.SAFETY_BOUND
            vertices1[1] -= self.tree_width / 2.0
            vertices2 = np.copy(near_row_vertice)
            vertices2[0] -= self.SAFETY_BOUND
            vertices2[1] += self.tree_width / 2.0
            vertices3 = np.copy(far_row_vertice)
            vertices3[0] += self.SAFETY_BOUND
            vertices3[1] += self.tree_width / 2.0
            vertices4 = np.copy(far_row_vertice)
            vertices4[0] += self.SAFETY_BOUND
            vertices4[1] -= self.tree_width / 2.0
            row_obs = np.vstack([vertices1, vertices2, vertices3, vertices4])
            # keep the decimal in 7 digits
            row_obs = np.round(row_obs, 7)
            row_obstacles.append(row_obs)

        # intermediate
        row_obstacles += intermediate_row_obstacles

        return row_obstacles

    def get_obstacle_tree_rows(self, start_pose, end_pose):
        larger_y = max(start_pose[1], end_pose[1])
        smaller_y = min(start_pose[1], end_pose[1])
        intermediate_rows_idxs = np.where(
            (self.map_tree_rows[:, 0, 1] > smaller_y)
            & (self.map_tree_rows[:, 0, 1] < larger_y)
        )[0]
        intermediate_rows_idxs = np.sort(intermediate_rows_idxs)
        intermediate_row_low_idx = intermediate_rows_idxs[0]
        intermediate_row_up_idx = intermediate_rows_idxs[-1]
        # neighboured rows
        # TODO: row width cannot be too small otherwise the constrain will fail!!
        row_obstacles = []
        if intermediate_row_low_idx == intermediate_row_up_idx:
            relevant_row_idx_low = max(intermediate_row_up_idx - 2, 0)
            relevant_row_idx_up = min(
                intermediate_row_up_idx + 2, len(self.map_tree_rows) - 1
            )
            for idx in range(relevant_row_idx_low, relevant_row_idx_up):
                near_row_vertice = np.copy(self.map_tree_rows[idx, 0])
                far_row_vertice = np.copy(self.map_tree_rows[idx, 1])
                # formulate it into a rectangle
                vertices1 = np.copy(near_row_vertice)
                vertices1[0] -= self.SAFETY_BOUND
                vertices1[1] -= self.tree_width / 2.0
                vertices2 = np.copy(near_row_vertice)
                vertices2[0] -= self.SAFETY_BOUND
                vertices2[1] += self.tree_width / 2.0
                vertices3 = np.copy(far_row_vertice)
                vertices3[0] += self.SAFETY_BOUND
                vertices3[1] += self.tree_width / 2.0
                vertices4 = np.copy(far_row_vertice)
                vertices4[0] += self.SAFETY_BOUND
                vertices4[1] -= self.tree_width / 2.0
                row_obs = np.vstack([vertices1, vertices2, vertices3, vertices4])
                # keep the decimal in 7 digits
                row_obs = np.round(row_obs, 7)
                row_obstacles.append(row_obs)
            return row_obstacles

        relevant_rows_low_idx = max(intermediate_row_low_idx - 2, 0)
        relevant_rows_up_idx = min(
            intermediate_row_up_idx + 3, len(self.map_tree_rows) - 1
        )

        row_obstacles = []
        relevant_row_idxs = list(range(relevant_rows_low_idx, relevant_rows_up_idx))
        for row in self.map_tree_rows[relevant_row_idxs]:
            # no extra on cap
            near_row_vertice = np.copy(row[0])
            far_row_vertice = np.copy(row[1])
            # far_row_vertice = np.copy((row[0] + row[1]) / 2)
            # far_row_vertice[0] = near_row_vertice[0] + 100

            vertices1 = np.copy(near_row_vertice)
            vertices1[0] -= self.SAFETY_BOUND
            vertices1[1] -= self.tree_width / 2.0
            vertices2 = np.copy(near_row_vertice)
            vertices2[0] -= self.SAFETY_BOUND
            vertices2[1] += self.tree_width / 2.0
            vertices3 = np.copy(far_row_vertice)
            vertices3[0] += self.SAFETY_BOUND
            vertices3[1] += self.tree_width / 2.0
            vertices4 = np.copy(far_row_vertice)
            vertices4[0] += self.SAFETY_BOUND
            vertices4[1] -= self.tree_width / 2.0
            row_obs = np.vstack([vertices1, vertices2, vertices3, vertices4])
            # keep the decimal in 7 digits
            # row_obs = np.round(row_obs, 7)
            row_obstacles.append(row_obs)
        # up_move_indexs = list(range(intermediate_row_low_idx, relevant_rows_up_idx + 1))
        # for row in self.map_tree_rows[up_move_indexs]:
        #     # no extra on cap
        #     near_row_vertice = np.copy(row[0])
        #     far_row_vertice = np.copy(row[1])

        #     vertices1 = np.copy(near_row_vertice)
        #     vertices1[0] -= self.SAFETY_BOUND
        #     vertices1[1] -= self.tree_width / 2.0
        #     vertices2 = np.copy(near_row_vertice)
        #     vertices2[0] -= self.SAFETY_BOUND
        #     vertices2[1] += self.MIN_ROW_WIDTH - self.tree_width / 2.0
        #     vertices3 = np.copy(far_row_vertice)
        #     vertices3[0] += self.SAFETY_BOUND
        #     vertices3[1] += self.MIN_ROW_WIDTH - self.tree_width / 2.0
        #     vertices4 = np.copy(far_row_vertice)
        #     vertices4[0] += self.SAFETY_BOUND
        #     vertices4[1] -= self.tree_width / 2.0
        #     row_obs = np.vstack([vertices1, vertices2, vertices3, vertices4])
        #     # keep the decimal in 7 digits
        #     # row_obs = np.round(row_obs, 7)
        #     row_obstacles.append(row_obs)

        # down_move_indexs = [relevant_rows_low_idx, intermediate_row_up_idx]
        # for row in self.map_tree_rows[down_move_indexs]:
        #     near_row_vertice = np.copy(row[0])
        #     far_row_vertice = np.copy(row[1])

        #     vertices1 = np.copy(near_row_vertice)
        #     vertices1[0] -= self.SAFETY_BOUND
        #     vertices1[1] -= self.MIN_ROW_WIDTH - self.tree_width / 2.0
        #     vertices2 = np.copy(near_row_vertice)
        #     vertices2[0] -= self.SAFETY_BOUND
        #     vertices2[1] += self.tree_width / 2.0
        #     vertices3 = np.copy(far_row_vertice)
        #     vertices3[0] += self.SAFETY_BOUND
        #     vertices3[1] += self.tree_width / 2.0
        #     vertices4 = np.copy(far_row_vertice)
        #     vertices4[0] += self.SAFETY_BOUND
        #     vertices4[1] -= self.MIN_ROW_WIDTH - self.tree_width / 2.0
        #     row_obs = np.vstack([vertices1, vertices2, vertices3, vertices4])
        #     # keep the decimal in 7 digits
        #     # row_obs = np.round(row_obs, 7)
        #     row_obstacles.append(row_obs)
        return row_obstacles

    def get_obstacles_for_OBCA(
        self,
        boundary_polys,
        row_polys,
        start_pose,
        end_pose,
        side,
        width=2,
        buffer_distance=1,
    ):
        # 1) near far low up
        obstacles = []
        index = 0 if side == self.NEAR_SIDE else 1

        if len(boundary_polys[index]) <= 1:
            obstacles.append(*boundary_polys[index])

        else:
            rectangle_obstacles = []
            pose_y_max = max(start_pose[1], end_pose[1])
            pose_y_min = min(start_pose[1], end_pose[1])
            for poly in boundary_polys[index]:
                poly_y_min = np.min(poly[:2, 1])
                poly_y_max = np.max(poly[:2, 1])
                if (
                    poly_y_max > pose_y_min - buffer_distance
                    and poly_y_min < pose_y_max + buffer_distance
                ):
                    rectangle_obstacles += [poly]

            # combine convex polygons
            contour_points = np.array(
                [obstacle[0, :] for obstacle in rectangle_obstacles]
            )
            contour_points = np.vstack([contour_points, rectangle_obstacles[-1][1, :]])
            dist_signed = -width if side == self.NEAR_SIDE else width
            direction_signed = 1 if side == self.NEAR_SIDE else -1
            y_dir = np.sign(contour_points[1][1] - contour_points[0][1])
            dist_signed *= y_dir
            direction_signed *= y_dir

            start_idx = 0
            end_idx = start_idx + 1
            current_poly = np.array(
                [
                    contour_points[start_idx],
                    contour_points[end_idx],
                ]
            )
            while end_idx < len(contour_points) - 1:
                first_pt = contour_points[end_idx - 1]
                second_pt = contour_points[end_idx]
                third_pt = contour_points[end_idx + 1]
                direction = point_side_of_line(first_pt, second_pt, third_pt)
                if direction == direction_signed:  # satisfy convex condition
                    current_poly = np.vstack([current_poly, third_pt])
                    end_idx += 1
                else:  # otherwise
                    point1, point2 = points_along_rectangles(
                        current_poly[0], current_poly[-1], dist_signed
                    )
                    current_poly = np.vstack([current_poly, point1, point2])
                    obstacles.append(current_poly)

                    start_idx = end_idx
                    end_idx = start_idx + 1
                    current_poly = np.array(
                        [
                            contour_points[start_idx],
                            contour_points[end_idx],
                        ]
                    )

            point1, point2 = points_along_rectangles(
                current_poly[0], current_poly[-1], dist_signed
            )
            current_poly = np.vstack([current_poly, point1, point2])
            obstacles.append(current_poly)

        # 2) up and low
        if start_pose[1] > end_pose[1]:
            obstacles += [poly for poly in boundary_polys[2]]
        else:
            obstacles += [poly for poly in boundary_polys[3]]
        return obstacles + row_polys
