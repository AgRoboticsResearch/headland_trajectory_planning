import numpy as np
import casadi as ca
from typing import List, Tuple, Dict, Any
from pypoman import compute_polytope_halfspaces

from car_model_obca import CarModel


def kinematic_model(wheel_base: float) -> ca.casadi.Function:
    """
    Kinematic Model:

    * state = [x, y, v, theta, steer_angle]
    * control = [accel, steer_rate]
    """
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    v = ca.SX.sym("v")
    theta = ca.SX.sym("theta")
    steer_angle = ca.SX.sym("steer_angle")
    state = ca.vertcat(x, y, v, theta, steer_angle)

    accel = ca.SX.sym("accel")
    steer_rate = ca.SX.sym("steer_rate")
    control = ca.vertcat(accel, steer_rate)

    # TODO: Add first-order dynamic to the accel and steer_rate control
    rhs = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        accel,
        v * ca.tan(steer_angle) / wheel_base,
        steer_rate,
    )

    return ca.Function("f", [state, control], [rhs])


class PolySet(object):
    """
    Polygons/Polytopes Set Class
    """

    def __init__(self, poly_list: List) -> None:
        if not isinstance(poly_list, list):
            raise Exception("[OBCA] The input to PolySet should be a list!")

        if len(poly_list) < 1:
            print("[OBCA] The input to PolySet is empty.")

        self.poly_list = poly_list
        self.edge_counts = []
        self.total_edges = 0
        for i in range(len(poly_list)):
            edge_count = len(self.poly_list[i])
            self.edge_counts.append(edge_count)
            self.total_edges += edge_count
        self.edge_counts = np.array(self.edge_counts, dtype=int)

    def __len__(self) -> int:
        return len(self.poly_list)

    def __getitem__(self, idx: int) -> Any:
        return self.poly_list[idx]


class OBCAOptimizer(object):
    """
    The OBCA Optimizer
    """

    DEFAULT_MAX_VELOCITY = 1.0  # m/s
    DEFAULT_MAX_ACCEL = 1.0  # m/s^2
    DEFAULT_MAX_STEER_RATE = 0.7  # rad/s
    DEFAULT_MIN_DISTANCE_TO_OBS = 0.1  # m

    def __init__(
        self,
        car: CarModel,
        obstacles: List,
        init_traj: np.ndarray,
        enable_aux: bool = True,
        init_control: np.ndarray = None,
        init_dual_var: List = None,
        dT: float = 0.2,
        Q: np.ndarray = np.diag([1, 1]),
        R: np.ndarray = np.diag([0.1, 0.1]),
        W: np.ndarray = np.diag([5, 0]),
        x_bound: List = [-ca.inf, ca.inf],
        y_bound: List = [-ca.inf, ca.inf],
        max_velocity: float = DEFAULT_MAX_VELOCITY,
        max_accel: float = DEFAULT_MAX_ACCEL,
        max_steer_rate: float = DEFAULT_MAX_STEER_RATE,
        min_dist_to_obs: float = DEFAULT_MIN_DISTANCE_TO_OBS,
    ) -> None:
        # initialize ocp parameters
        self.variables = []  # decision variables
        self.constraints = []  # constraints
        self.lbx = []  # states lower bounds
        self.ubx = []  # states upper bounds
        self.lbg = []  # constraints lower bounds
        self.ubg = []  # constraints upper bounds
        self.x0 = []  # initial guess
        self.n_controls = 2  # control numbers
        self.n_states = 5  # state variable depending on the model
        self.dT = dT  # time interval

        # enable time_scale optimization
        self.enable_time_opt = False if W[1, 1] == 0 else True

        # set minimum distance to obstacles # TODO: make it obstacle dependent
        self.MIN_DISTANCE_TO_OBS = self.DEFAULT_MIN_DISTANCE_TO_OBS
        if min_dist_to_obs is not None:
            if min_dist_to_obs < 0:
                print(
                    "[OBCA] Minimum distance to obstacles cannot be negative! Use default value."
                )
            else:
                self.MIN_DISTANCE_TO_OBS = min_dist_to_obs

        # set vehicle parameters
        self.set_vehicle_param(car, max_velocity, max_accel, max_steer_rate)
        # set x and y boundaries
        self.set_x_y_boundary(x_bound, y_bound)
        # Generate control objects
        self.generate_control_objects(car, enable_aux)
        # Generate obstacles
        self.generate_obstacles(obstacles)
        # Set initial guess
        self.set_initial_guess(init_traj, init_control, init_dual_var)
        # Generate decision variables
        self.generate_variables()
        # Generate constraints
        self.generate_constraints()
        # Generate objective function
        self.generate_objective(Q, R, W)

        print("[OBCA] The solver has been successfully initialized!")

    def set_vehicle_param(
        self, car: CarModel, max_velocity: float, max_accel: float, max_steer_rate: float
    ) -> None:
        if car.WHEEL_BASE < 0:
            raise Exception("[OBCA] Wheelbase length should be a positive number!")
        else:
            self.WHEEL_BASE = car.WHEEL_BASE

        self.MAX_STEER = abs(car.MAX_STEER)
        self.MAX_VELOCITY = abs(max_velocity)
        self.MAX_ACCEL = abs(max_accel)
        self.MAX_STEER_RATE = abs(max_steer_rate)

        # print("[OBCA] Wheelbase length = ", self.WHEEL_BASE)
        # print("[OBCA] Maximum steer angle = ", self.MAX_STEER)
        # print("[OBCA] Maximum steer rate = ", self.MAX_STEER_RATE)
        # print("[OBCA] Maximum velocity = ", self.MAX_VELOCITY)
        # print("[OBCA] Maximum acceleration = ", self.MAX_ACCEL)

    def generate_control_objects(self, car: CarModel, enable_aux: bool) -> None:
        # create control objects halfspaces
        self.Gs, self.gs = self.get_polytopes_for_control_objects(car, enable_aux)

    def get_polytopes_for_control_objects(
        self, car: CarModel, enable_aux: bool
    ) -> Tuple[List, List]:
        # get control object polyset
        control_objects = [car.car_poly]
        if enable_aux:
            if len(car.aux_polys) != 0:
                control_objects += car.aux_polys
            else:
                print("[OBCA] Implements not found! Use empty car model by default.")

        control_objects_np = [
            np.array(poly.exterior.xy).T[:-1] for poly in control_objects
        ]
        self.control_objects = PolySet(control_objects_np)

        # calculate halfspaces
        Gs, gs = [], []
        for poly in control_objects:
            poly_vertices = np.array(poly.exterior.xy).T
            poly_vertices = poly_vertices[:-1]  # remove repeated vertix
            poly_vertices = np.round(poly_vertices, 7)
            G, g = compute_polytope_halfspaces(poly_vertices)
            G, g = np.round(G, 7), np.round(g, 7)  # for numerical stability
            Gs.append(G)
            gs.append(g)

        return Gs, gs

    def generate_obstacles(self, obstacles: List) -> None:
        # get obstacles polyset
        self.obstacles = PolySet(obstacles)
        # calculate halfspaces
        self.As, self.bs = [], []
        for poly in obstacles:
            poly = np.round(poly, 7)
            A, b = compute_polytope_halfspaces(poly)
            A, b = np.round(A, 7), np.round(b, 7)  # for numerical stability
            self.As.append(A)
            self.bs.append(b)
            # print("A: ", A)
            # print("b: ", b)

    def set_init_state(self, init_state: np.ndarray) -> None:
        if init_state is None:
            raise Exception("[OBCA] Init state can not be Empty!")
        self.init_state = ca.SX(init_state)  # x, y, v, yaw, steer
        print("[OBCA] Init State = ", init_state)

    def set_end_state(self, end_state: np.ndarray) -> None:
        if end_state is None:
            raise Exception("[OBCA] End state can not be Empty!")
        self.end_state = ca.SX(end_state)  # x, y, v, yaw, steer
        print("[OBCA] End State = ", end_state)

    def set_x_y_boundary(self, x_bound: List, y_bound: List) -> None:
        if x_bound[1] < x_bound[0]:
            raise Exception("[OBCA] The x_bound is infeasible!")
        if y_bound[1] < y_bound[0]:
            raise Exception("[OBCA] The y_bound is infeasible!")
        self.x_bound = x_bound
        self.y_bound = y_bound
        # print("[OBCA] X boundary = ", self.x_bound)
        # print("[OBCA] Y boundary = ", self.y_bound)

    def set_initial_guess(
        self, init_traj: np.ndarray, init_control: np.ndarray, init_dual_var: List
    ) -> None:
        # prediction steps
        self.N = len(init_traj)
        if self.N < 1:
            raise Exception("[OBCA] Initial guess is empty!")
        print("[OBCA] Prediction steps: ", self.N)

        # prediction horizon
        if not self.enable_time_opt:
            self.horizon = (self.N - 1) * self.dT
            print("[OBCA] Prediction horizon: ", self.horizon)

        # set init and terminal state
        self.set_init_state(init_traj[0, :])
        self.set_end_state(init_traj[-1, :])

        """
        decision variable = [state, control, mu, lambda, *time_scale, slack]
        * is optional
        """
        # initialize states
        for state in init_traj:
            self.x0 += [[state[i]] for i in range(self.n_states)]

        # initialize control inputs
        if init_control is None:
            self.x0 += [[0] * (self.n_controls * (self.N - 1))]
        else:
            if (
                init_control.shape[0] != self.N - 1
                or init_control.shape[1] != self.n_controls
            ):
                raise Exception("[OBCA] The control input dimension does not match!")
            for control in init_control:
                self.x0 += [[control[i]] for i in range(self.n_controls)]

        # initializes dual variables (mu + lambda)
        mu_count = self.control_objects.total_edges * len(self.obstacles)
        lambda_count = self.obstacles.total_edges * len(self.control_objects)
        if init_dual_var is None:
            self.x0 += [[0.1] * (mu_count + lambda_count) * (self.N)]
        else:
            init_mu, init_lambda = init_dual_var[0], init_dual_var[1]
            if (
                init_mu.shape[0] != self.N
                or init_mu.shape[1] != mu_count
                or init_lambda.shape[0] != self.N
                or init_lambda.shape[1] != lambda_count
            ):
                raise Exception("[OBCA] The dual variable dimension does not match!")
            for mu in init_mu:
                self.x0 += [val for val in mu]
            for lamda in init_lambda:
                self.x0 += [val for val in lamda]

        # initialize time scales
        if self.enable_time_opt:
            self.x0 += [1] * (self.N - 1)

        # initialize slack variables
        self.x0 += [[0] * self.n_states]

    def generate_variables(self) -> None:
        # states
        self.X = ca.SX.sym("X", self.n_states, self.N)
        for i in range(self.N):
            self.variables += [self.X[:, i]]
            self.lbx += [
                self.x_bound[0],
                self.y_bound[0],
                -self.MAX_VELOCITY,
                -2 * ca.pi,
                -self.MAX_STEER,
            ]
            self.ubx += [
                self.x_bound[1],
                self.y_bound[1],
                self.MAX_VELOCITY,
                2 * ca.pi,
                self.MAX_STEER,
            ]

        # control inputs
        self.U = ca.SX.sym("U", self.n_controls, self.N - 1)
        for i in range(self.N - 1):
            self.variables += [self.U[:, i]]
            self.lbx += [-self.MAX_ACCEL, -self.MAX_STEER_RATE]
            self.ubx += [self.MAX_ACCEL, self.MAX_STEER_RATE]

        # dual variables (mu + lambda)
        self.MU = ca.SX.sym(
            "MU", self.control_objects.total_edges * len(self.obstacles), self.N
        )
        self.LAMBDA = ca.SX.sym(
            "LAMBDA", self.obstacles.total_edges * len(self.control_objects), self.N
        )
        for i in range(self.N):
            start_idx = 0  # mu
            for _ in range(len(self.obstacles)):
                for edge_count in self.control_objects.edge_counts:
                    self.variables += [self.MU[start_idx : start_idx + edge_count, i]]
                    self.lbx += [0] * edge_count
                    self.ubx += [ca.inf] * edge_count
                    start_idx += edge_count
        for i in range(self.N):
            start_idx = 0  # lambda
            for _ in range(len(self.control_objects)):
                for edge_count in self.obstacles.edge_counts:
                    self.variables += [self.LAMBDA[start_idx : start_idx + edge_count, i]]
                    self.lbx += [0] * edge_count
                    self.ubx += [ca.inf] * edge_count
                    start_idx += edge_count

        # time scales
        if self.enable_time_opt:
            self.time_scale = ca.SX.sym("tau", self.N - 1)
            self.variables += [self.time_scale]
            self.lbx += [0.05 / self.dT] * (self.N - 1)
            self.ubx += [1] * (self.N - 1)

        # slack for terminal constraint
        self.slack_e = ca.SX.sym("slack_e", self.n_states, 1)
        self.variables += [self.slack_e]
        self.lbx += [-ca.inf] * self.n_states
        self.ubx += [ca.inf] * self.n_states

    def generate_constraints(self) -> None:
        # initial state constraint
        self.constraints += [self.X[:, 0] - self.init_state]
        self.lbg += [0 for _ in range(self.n_states)]
        self.ubg += [0 for _ in range(self.n_states)]

        # kinematic model constraints
        func = kinematic_model(self.WHEEL_BASE)
        for i in range(self.N - 1):
            state = self.X[:, i]
            control = self.U[:, i]
            if self.enable_time_opt:  # if optimize time scales
                dT = self.dT * self.time_scale[i]
                f_value = func(state + 0.5 * dT * func(state, control), control)
                state_next_euler = state + dT * f_value
            else:
                f_value = func(state, control)
                state_next_euler = state + self.dT * f_value
            state_next = self.X[:, i + 1]
            self.constraints += [state_next - state_next_euler]
            self.lbg += [0 for _ in range(self.n_states)]
            self.ubg += [0 for _ in range(self.n_states)]

        # terminal state constraint
        # self.constraints += [self.X[:, -1] - self.end_state]
        self.constraints += [self.X[:, -1] - self.end_state + self.slack_e]
        self.lbg += [0 for _ in range(self.n_states)]
        self.ubg += [0 for _ in range(self.n_states)]

        # collision avoidance constraints
        for i in range(self.N):
            state = self.X[:, i]
            heading = state[3]
            x, y = state[0], state[1]
            # calculate the rotaiton and translation vector
            trans = ca.vertcat(x, y)
            rot = ca.SX(2, 2)
            rot[0, 0] = ca.cos(heading)
            rot[0, 1] = -ca.sin(heading)
            rot[1, 0] = ca.sin(heading)
            rot[1, 1] = ca.cos(heading)

            for m in range(len(self.obstacles)):
                A, b = self.As[m], self.bs[m]
                for n in range(len(self.control_objects)):
                    G, g = self.Gs[n], self.gs[n]
                    # lambda
                    lamb_start_idx = (
                        np.sum(self.obstacles.edge_counts[:m])
                        + n * self.obstacles.total_edges
                    )
                    lamb_end_idx = lamb_start_idx + self.obstacles.edge_counts[m]
                    lamb = ca.vertcat(self.LAMBDA[lamb_start_idx:lamb_end_idx, i])
                    # mu
                    mu_start_idx = (
                        np.sum(self.control_objects.edge_counts[:n])
                        + m * self.control_objects.total_edges
                    )
                    mu_end_idx = mu_start_idx + self.control_objects.edge_counts[n]
                    mu = ca.vertcat(self.MU[mu_start_idx:mu_end_idx, i])
                    # signed distance constraints (duality)
                    self.constraints += [ca.dot(A.T @ lamb, A.T @ lamb)]
                    self.lbg += [0]
                    self.ubg += [1]
                    self.constraints += [G.T @ mu + (rot.T @ A.T) @ lamb]
                    self.lbg += [0, 0]
                    self.ubg += [0, 0]
                    self.constraints += [(-ca.dot(g, mu) + ca.dot(A @ trans - b, lamb))]
                    self.lbg += [self.MIN_DISTANCE_TO_OBS]
                    self.ubg += [ca.inf]

    def generate_objective(self, Q: np.ndarray, R: np.ndarray, W: np.ndarray) -> None:
        self.weight = {}
        # Weight Q matrix (control effort)
        if len(Q) != self.n_controls:
            raise Exception("[OBCA] Weight_Q dimension does not match!")
        self.weight["Q"] = Q
        Q = ca.SX(Q.tolist())

        # Weight R matrix (jerk)
        if len(R) != self.n_controls:
            raise Exception("[OBCA] Weight_R dimension does not match!")
        self.weight["R"] = R
        R = ca.SX(R.tolist())

        # Weight W matrix (path length, total time)
        if len(W) != 2:
            raise Exception("[OBCA] Weight_W dimension does not match!")
        self.weight["W"] = W
        W = ca.SX(W.tolist())

        self.objective = 0.0
        for i in range(self.N - 1):
            # if enable time optimization
            if self.enable_time_opt:
                dT = self.dT * self.time_scale[i]
                self.objective += dT * W[1, 1]
            else:
                dT = self.dT

            # 1) control effort penalty
            control = self.U[:, i]
            self.objective += control.T @ Q @ control

            # 2) jerk penalty
            if i < self.N - 2:
                speed_jerk = (self.U[0, i + 1] - self.U[0, i]) / dT
                steer_jerk = (self.U[1, i + 1] - self.U[1, i]) / dT
                jerk = ca.vertcat(speed_jerk, steer_jerk)
                self.objective += jerk.T @ R @ jerk

            # 3) path length penalty
            dist = self.X[2, i] * dT
            self.objective += (dist) ** 2 * W[0, 0]

        # 4) slack penalty
        S = ca.SX((5000 * np.identity(self.n_states)).tolist())
        self.objective += self.slack_e.T @ S @ self.slack_e

    def solve(self, max_cpu_time=20, verbose: bool = False) -> Tuple[bool, Dict]:
        nlp_prob = {
            "f": self.objective,
            "x": ca.vertcat(*self.variables),
            "g": ca.vertcat(*self.constraints),
        }
        opts = {
            "print_time": 1 if verbose else 0,
            "ipopt": {
                "print_level": 5 if verbose else 0,
                "sb": "yes",
                "max_cpu_time": max_cpu_time,
            },
        }
        solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
        print("[OBCA] Number of decision variables: ", nlp_prob["x"].size(1))
        print(
            "[OBCA] Number of equality constraints: ",
            np.sum(lb == ub for lb, ub in zip(self.lbg, self.ubg)),
        )
        print(
            "[OBCA] Number of inequality constraints: ",
            np.sum(lb != ub for lb, ub in zip(self.lbg, self.ubg)),
        )

        # get solution
        result = solver(
            x0=ca.vertcat(*self.x0),
            lbx=self.lbx,
            ubx=self.ubx,
            ubg=self.ubg,
            lbg=self.lbg,
        )
        variable_opt = result["x"]

        # check solution status
        success = solver.stats()["success"]
        print("[OBCA] EXIT: ", solver.stats()["return_status"])

        # optimial states
        x_opt = variable_opt[0 : self.n_states * (self.N) : self.n_states]
        y_opt = variable_opt[1 : self.n_states * (self.N) : self.n_states]
        v_opt = variable_opt[2 : self.n_states * (self.N) : self.n_states]
        theta_opt = variable_opt[3 : self.n_states * (self.N) : self.n_states]
        steer_angle_opt = variable_opt[4 : self.n_states * (self.N) : self.n_states]
        # optimial control inputs
        accel_opt = variable_opt[
            self.n_states * self.N : self.n_states * self.N
            + self.n_controls * (self.N - 1) : self.n_controls
        ]
        steer_rate_opt = variable_opt[
            self.n_states * self.N
            + 1 : self.n_states * self.N
            + self.n_controls * (self.N - 1) : self.n_controls
        ]
        # optimal dual variables
        mu_count = self.control_objects.total_edges * len(self.obstacles)
        lambda_count = self.obstacles.total_edges * len(self.control_objects)
        mu_opt = variable_opt[
            self.n_states * self.N
            + self.n_controls * (self.N - 1) : self.n_states * self.N
            + self.n_controls * (self.N - 1)
            + mu_count * self.N
        ]
        lambda_opt = variable_opt[
            self.n_states * self.N
            + self.n_controls * (self.N - 1)
            + mu_count * self.N : self.n_states * self.N
            + self.n_controls * (self.N - 1)
            + (mu_count + lambda_count) * (self.N)
        ]
        # optimal time scales
        if self.enable_time_opt:
            time_scale_opt = variable_opt[-5 - (self.N - 1) : -5]
        else:
            time_scale_opt = ca.DM.ones(self.N - 1)
        # optimal slack variables
        slack_opt = variable_opt[-5:]

        solution = {
            "dT": self.dT,
            "weight": self.weight,
            "x_opt": np.array(x_opt.elements()),
            "y_opt": np.array(y_opt.elements()),
            "v_opt": np.array(v_opt.elements()),
            "theta_opt": np.array(theta_opt.elements()),
            "steer_angle_opt": np.array(steer_angle_opt.elements()),
            "accel_opt": np.array(accel_opt.elements()),
            "steer_rate_opt": np.array(steer_rate_opt.elements()),
            "mu_opt": np.array(mu_opt.elements()).reshape(self.N, mu_count),
            "lambda_opt": np.array(lambda_opt.elements()).reshape(self.N, lambda_count),
            "time_scale_opt": np.array(time_scale_opt.elements()),
            "slack_opt": np.array(slack_opt.elements()),
            "objective": result["f"].elements()[0],
        }

        return success, solution

    @staticmethod
    def show_slack(solution: Dict) -> None:
        slack_opt = solution["slack_opt"]
        print("x_slack = ", slack_opt[0])
        print("y_slack = ", slack_opt[1])
        print("v_slack = ", slack_opt[2])
        print("yaw_slack = ", slack_opt[3])
        print("steer_slack = ", slack_opt[4])

    @staticmethod
    def show_cost(solution: Dict) -> None:
        objective = solution["objective"]
        dT = solution["dT"]
        v_opt = solution["v_opt"]
        accel_opt = solution["accel_opt"]
        steerate_opt = solution["steer_rate_opt"]
        time_scale_opt = solution["time_scale_opt"]
        slack_opt = solution["slack_opt"]
        Q = solution["weight"]["Q"]
        R = solution["weight"]["R"]
        W = solution["weight"]["W"]

        accel_cost = (accel_opt**2).sum() * Q[0, 0]
        steerate_cost = (steerate_opt**2).sum() * Q[1, 1]
        control_effort_cost = accel_cost + steerate_cost

        speed_jerk = (
            (np.diff(accel_opt) / (dT * time_scale_opt[:-1])) ** 2 * R[0, 0]
        ).sum()
        steer_jerk = (
            (np.diff(steerate_opt) / (dT * time_scale_opt[:-1])) ** 2 * R[1, 1]
        ).sum()
        jerk_cost = speed_jerk + steer_jerk

        dist = (np.abs(v_opt[:-1]) * (dT * time_scale_opt)) ** 2
        dist_cost = dist.sum() * W[0, 0]

        total_time = time_scale_opt * dT
        total_time_cost = total_time.sum() * W[1, 1]

        slack_cost = (slack_opt**2 * 1000).sum()

        print("Total cost: ", objective)
        print("control effort cost: ", control_effort_cost)
        print("jerk cost: ", jerk_cost)
        print("path length cost: ", dist_cost)
        print("total time cost: ", total_time_cost)
        print("slack cost: ", slack_cost)
        # print(control_effort_cost + jerk_cost + dist_cost + total_time_cost + slack_cost)
