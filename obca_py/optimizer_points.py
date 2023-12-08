from car_model_obca import CarModel
import casadi as ca
from pypoman import compute_polytope_halfspaces
import numpy as np
from shapely.geometry import MultiPoint


class OBCAOptimizer:
    MAX_VELOCITY = 1
    MAX_ACCEL = 1
    MAX_STEER_RATE = 0.7
    MIN_DISTANCE_TO_OBS = 0.1

    def __init__(self, car: CarModel = CarModel(), dT=0.2) -> None:
        self.v_car = car
        self.dT = dT
        # control numbers
        self.n_controls = 2
        # state variable depending on the model
        self.n_states = 5

        self.constrains = []
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self.variable = []
        self.N = 0
        self.x0 = []
        self.obstacles = []

        self.vertices = self.get_vehicle_vertices(self.v_car)
        print(self.vertices)

    def get_vehicle_vertices(self, car: CarModel, convex_hull=True):

        vertices = np.array([[0, 0]])
        for poly in [car.car_poly] + car.aux_polys:
            poly_vertices = np.array(poly.exterior.xy).T
            poly_vertices = poly_vertices[:-1]
            if len(vertices) == 0:
                vertices = poly_vertices
            else:
                vertices = np.vstack([vertices, poly_vertices])
        if convex_hull:
            points = MultiPoint(vertices)
            convex_hull = points.convex_hull
            vertices = np.array(convex_hull.exterior.coords)

        return vertices[:-1]

    def initialize_manual(
        self,
        init_guess_path,
        obs,
        min_x=-9999999,
        min_y=-9999999,
        max_x=9999999,
        max_y=9999999,
        init_control=None,
        init_dual_var=None,
    ):
        self.init_state = ca.SX(
            [
                init_guess_path[0, 0],
                init_guess_path[0, 1],
                init_guess_path[0, 2],
                init_guess_path[0, 3],
                init_guess_path[0, 4],
            ]
        )
        self.end_state = ca.SX(
            [
                init_guess_path[-1, 0],
                init_guess_path[-1, 1],
                init_guess_path[-1, 2],
                init_guess_path[-1, 3],
                init_guess_path[-1, 4],
            ]
        )
        self.N = len(init_guess_path)

        self.obstacles = obs
        for state in init_guess_path:
            self.x0 += [[state[i]] for i in range(self.n_states)]
        if init_control is None:
            self.x0 += [[0] * (self.n_controls * (self.N - 1))]
        else:
            for control in init_control:
                self.x0 += [[control[i]] for i in range(self.n_controls)]

        self.obs_dual_ns = self.get_dual_variable_ns()
        self.obs_dual_n_all = np.sum(self.obs_dual_ns)
        # dual_variable_num = self.n_dual_variable * (self.N) * len(obs)
        dual_variable_num = self.obs_dual_n_all * (self.N)
        self.x0 += [[0.1] * dual_variable_num]
        self.ref_state = init_guess_path
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.solution_found = False
        print(
            "number of constraints for obstacle free: ",
            self.N * len(self.obstacles) * 2,
            "number of variables: ",
            (self.n_states + self.N + dual_variable_num),
        )

    def get_dual_variable_ns(self):
        obs_dual_ns = []
        for obs in self.obstacles:
            dual_n = len(obs)
            obs_dual_ns.append(dual_n)

        return obs_dual_ns

    def build_model(self) -> bool:
        if self.N < 1:
            print("empty init guess")
            return False
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        v = ca.SX.sym("v")
        theta = ca.SX.sym("theta")
        steering = ca.SX.sym("steering")
        a = ca.SX.sym("a")
        steering_rate = ca.SX.sym("steering_rate")
        self.state = ca.vertcat(ca.vertcat(x, y, v, theta), steering)
        self.control = ca.vertcat(a, steering_rate)
        self.rhs = ca.vertcat(
            ca.vertcat(
                v * ca.cos(theta),
                v * ca.sin(theta),
                a,
                v * ca.tan(steering) / self.v_car.WHEEL_BASE,
            ),
            steering_rate,
        )

        self.f = ca.Function("f", [self.state, self.control], [self.rhs])
        self.X = ca.SX.sym("X", self.n_states, self.N)
        self.U = ca.SX.sym("U", self.n_controls, self.N - 1)

        # TODO: change these two varaibles dim accordingly
        # depend on the obstacles dim
        # self.LAMBDA = ca.SX.sym(
        #     "LAMBDA",
        #     self.n_dual_variable,
        #     self.N * len(self.obstacles),
        # )
        self.LAMBDA = ca.SX.sym("LAMBDA", self.obs_dual_n_all * self.N, 1)
        self.obj = 0

        return True

    def solve(self):
        nlp_prob = {
            "f": self.obj,
            "x": ca.vertcat(*self.variable),
            "g": ca.vertcat(*self.constrains),
        }
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
        self.solution_found = False
        # solver = ca.nlpsol("solver", "ipopt", nlp_prob)
        sol = solver(
            x0=ca.vertcat(*self.x0),
            lbx=self.lbx,
            ubx=self.ubx,
            ubg=self.ubg,
            lbg=self.lbg,
        )
        u_opt = sol["x"]

        self.x_opt = u_opt[0 : self.n_states * (self.N) : self.n_states]
        self.y_opt = u_opt[1 : self.n_states * (self.N) : self.n_states]
        self.v_opt = u_opt[2 : self.n_states * (self.N) : self.n_states]
        self.theta_opt = u_opt[3 : self.n_states * (self.N) : self.n_states]
        self.steer_opt = u_opt[4 : self.n_states * (self.N) : self.n_states]
        self.a_opt = u_opt[
            self.n_states * (self.N) : self.n_states * (self.N)
            + self.n_controls * (self.N - 1) : self.n_controls
        ]
        self.steerate_opt = u_opt[
            self.n_states * (self.N)
            + 1 : self.n_states * (self.N)
            + self.n_controls * (self.N - 1) : self.n_controls
        ]
        if len(self.x_opt.elements()) > 0:
            self.solution_found = True

    def generate_object(self, r, q):
        R = ca.SX(r)
        Q = ca.SX(q)
        for i in range(self.N - 1):
            # reference error
            # st = self.X[:, i]
            # ref_st = self.x0[i]
            # error = st - ref_st
            # self.obj += error.T @ Q @ error

            # control item
            # con = self.U[:, i]
            # self.obj += con.T @ R @ con
            # # control change changes
            if i < self.N - 2:
                delta_steer = self.U[1, i + 1] - self.U[1, i]
                delta_v = self.U[0, i + 1] - self.U[0, i]
                self.obj += delta_steer * delta_steer + delta_v * delta_v

            # gear switch
            # self.obj += (
            #     -(ca.sign(self.X[2, i + 1] * self.X[2, i]) - 1)
            #     * ca.fabs(ca.sign(self.X[2, i + 1]))
            #     * ca.fabs(ca.sign(self.X[2, i]))
            #     * 1000
            # )
            # self.obj += ca.fabs((ca.sign(self.X[2, i + 1]) - ca.sign(self.X[2, i])))

            # path length item
            self.obj += (self.X[2, i] * self.dT) ** 2 * 20

    def generate_variable(self):
        for i in range(self.N):
            self.variable += [self.X[:, i]]
            self.lbx += [
                self.min_x,
                self.min_y,
                -self.MAX_VELOCITY,
                -2 * ca.pi,
                -self.v_car.MAX_STEER,
            ]
            self.ubx += [
                self.max_x,
                self.max_y,
                self.MAX_VELOCITY,
                2 * ca.pi,
                self.v_car.MAX_STEER,
            ]
        # constraints for inputs
        for i in range(self.N - 1):
            self.variable += [self.U[:, i]]
            self.lbx += [-self.MAX_ACCEL, -self.MAX_STEER_RATE]
            self.ubx += [self.MAX_ACCEL, self.MAX_STEER_RATE]

        # duality variables need to be larger than zero
        # for i in range(len(self.obstacles) * self.N):
        #     self.variable += [self.LAMBDA[:, i]]
        #     self.lbx += [0.0, 0.0, 0.0, 0.0]
        #     self.ubx += [100000, 100000, 100000, 100000]
        for i in range(self.obs_dual_n_all * self.N):
            self.variable += [self.LAMBDA[i]]
            self.lbx += [0.0]
            self.ubx += [100000]

    def generate_constrain(self):
        # initial state constraint
        self.constrains += [self.X[:, 0] - self.init_state]
        self.lbg += [0, 0, 0, 0, 0]
        self.ubg += [0, 0, 0, 0, 0]
        # kinematic model constraints
        for i in range(self.N - 1):
            st = self.X[:, i]
            con = self.U[:, i]
            f_value = self.f(st, con)
            st_next_euler = st + self.dT * f_value
            st_next = self.X[:, i + 1]
            self.constrains += [st_next - st_next_euler]
            self.lbg += [0, 0, 0, 0, 0]
            self.ubg += [0, 0, 0, 0, 0]
        # final state constraint
        self.constrains += [self.X[:, -1] - self.end_state]
        # x_opt, y_opt, v_opt, heading_opt, steer_opt
        # self.lbg += [-0.05, -0.05, 0.0, -0.02, -0.1]
        # self.ubg += [0.05, 0.05, 0.0, 0.02, 0.1]
        self.lbg += [-0.0, -0.0, 0.0, -0.0, -0.0]
        self.ubg += [0.0, 0.0, 0.0, 0.0, 0.0]

        # obstacle constraints

        for j, obstacle in enumerate(self.obstacles):
            A, b = compute_polytope_halfspaces(obstacle)
            obs_dual_n = len(obstacle)

            if j == 0:
                obs_dual_counted = 0
            else:
                obs_dual_counted += self.obs_dual_ns[j - 1]
            # print("obs_dual_counted: ", obs_dual_counted)
            for i in range(self.N):
                st = self.X[:, i]
                heading = st[3]
                # center of rear wheel
                x = st[0]
                y = st[1]
                # calculate the rotaiton and translation vector
                t = ca.vertcat(x, y)
                r = ca.SX(2, 2)
                r[0, 0] = ca.cos(heading)
                r[0, 1] = -ca.sin(heading)
                r[1, 0] = ca.sin(heading)
                r[1, 1] = ca.cos(heading)

                start_idx = self.N * (obs_dual_counted) + i * obs_dual_n
                end_idx = start_idx + obs_dual_n
                lamb = ca.vertcat(self.LAMBDA[start_idx:end_idx, 0])

                # for debug
                # if i == self.N - 1 or i == 0:
                #     print(
                #         start_idx,
                #         end_idx,
                #         lamb,
                #     )

                for k, vertice in enumerate(self.vertices):
                    vertice = np.copy(vertice)[np.newaxis, :].T
                    vertice_t = r @ vertice + t

                    self.constrains += [ca.dot(A.T @ lamb, A.T @ lamb)]
                    self.lbg += [0]
                    self.ubg += [1]

                    self.constrains += [(ca.dot(A @ vertice_t - b, lamb))]
                    self.lbg += [self.MIN_DISTANCE_TO_OBS]
                    self.ubg += [100000]
