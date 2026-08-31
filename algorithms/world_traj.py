import numpy as np
from math import factorial
from .graph_search import graph_search


class WorldTraj(object):
    # STUDENT CODE HERE

    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)
        """
        # Path planning parameters.
        self.resolution = np.array([0.25, 0.25, 0.25]) # m
        self.margins    = (0.45, 0.38, 0.30, 0.26)     # m, tried largest first
        self.seg_max    = 12.0                         # m

        # Speed limits.
        self.v_max = 6.00 # m/s
        self.v_nom = 4.20 # m/s, seed speed for the initial time allocation

        # Vehicle constants.
        self.mass            = 0.030    # kg
        self.inertia         = (1.43e-5, 1.43e-5, 2.89e-5) # kg*m^2
        self.arm_length      = 0.046    # m
        self.k_thrust        = 2.3e-8   # N/(rad/s)**2
        self.k_drag          = 7.8e-11  # N*m/(rad/s)**2
        self.rotor_speed_max = 2500.0   # rad/s
        self.actuator_frac   = 0.88
        self.g               = 9.81     # m/s^2

        # Rotor allocation and per-rotor force limit.
        l, gamma = self.arm_length, self.k_drag / self.k_thrust
        self.alloc_inv = np.linalg.inv(np.array([[1, 1, 1, 1],
                                                 [0, l, 0, -l],
                                                 [-l, 0, l, 0],
                                                 [gamma, -gamma, gamma, -gamma]]))
        self.f_rotor_max = (self.k_thrust * self.rotor_speed_max**2
                            * self.actuator_frac)

        # Polynomial basis and snap cost weights.
        self.n_coef  = 8
        self.basis_0 = np.array([self._basis(0.0, n) for n in range(self.n_coef)])
        self.basis_1 = np.array([self._basis(1.0, n) for n in range(self.n_coef)])
        self.Q_snap  = self._snap_gram(self.n_coef)

        # Trajectory refinement loops.
        self.dt_min         = 0.15 # s
        self.n_time_iters   = 20   # gradient steps on the time ratios
        self.n_insert_iters = 8
        self.check_dt       = 0.02 # s, sampling step for collision checking

        # Dense path and sparse waypoints.
        self.path = self._plan_path(world, start, goal)
        self.points, keep_idx = self._prune_waypoints(world, self.path)

        # Smoothed trajectory.
        self._refine_waypoints(world, keep_idx)


    def _plan_path(self, world, start, goal):
        """ Dense path at the largest planning margin that still admits one. """
        for margin in self.margins:
            self.margin = margin
            path, _ = graph_search(world, self.resolution, self.margin,
                                   start, goal, astar=True)
            if path is not None:
                return path

        # Fallback to stationary path.
        return np.vstack((start, start))


    def _refine_waypoints(self, world, keep_idx):
        """ Split colliding segments until the smoothed trajectory is clear. """
        # Initial fit.
        self.t_waypoints, self.coeffs = self._build(self.points)
        bad_segs = self._colliding_segments(world)

        # Iterative midpoint insertion.
        for _ in range(self.n_insert_iters):
            if not bad_segs:
                break
            inserted = self._insert_midpoints(keep_idx, bad_segs)
            if inserted is None:
                break
            self.points, keep_idx = inserted
            self.t_waypoints, self.coeffs = self._build(self.points)
            bad_segs = self._colliding_segments(world)

        # Dense-path fallback.
        if bad_segs and self.path.shape[0] > 2:
            self.points = self.path
            self.t_waypoints, self.coeffs = self._build(self.points)


    def _prune_waypoints(self, world, path):
        """ Greedy line-of-sight pruning; also returns dense-path indices. """
        if path.shape[0] <= 2:
            return path, list(range(path.shape[0]))

        # Farthest collision-free hop from each kept node.
        keep = [0]
        i = 0
        while i < path.shape[0] - 1:
            j_best = i + 1
            for j in range(path.shape[0] - 1, i, -1):
                seg = np.vstack((path[i], path[j]))
                if len(world.path_collisions(seg, self.margin)) == 0:
                    j_best = j
                    break
            keep.append(j_best)
            i = j_best

        keep = self._cap_spacing(path, keep)
        return path[keep], keep


    def _cap_spacing(self, path, keep):
        """ Subdivide any hop longer than seg_max with dense-path nodes. """
        capped = [keep[0]]
        for idx_a, idx_b in zip(keep[:-1], keep[1:]):
            # Evenly spaced dense-path nodes inside the hop.
            n_sub = int(np.ceil(np.linalg.norm(path[idx_b] - path[idx_a]) / self.seg_max))
            for k in range(1, n_sub):
                idx_sub = idx_a + int(round((idx_b - idx_a) * k / n_sub))
                if capped[-1] < idx_sub < idx_b:
                    capped.append(idx_sub)
            capped.append(idx_b)
        return capped


    def _insert_midpoints(self, keep_idx, bad_segs):
        """ Reinsert a dense-path node inside every segment that clipped an obstacle. """
        new_idx = []
        added = False
        for i, idx in enumerate(keep_idx[:-1]):
            new_idx.append(idx)
            if i not in bad_segs:
                continue
            idx_mid = (idx + keep_idx[i+1]) // 2
            if idx < idx_mid < keep_idx[i+1]:
                new_idx.append(idx_mid)
                added = True
        new_idx.append(keep_idx[-1])

        if not added:
            return None
        return self.path[new_idx], new_idx


    def _colliding_segments(self, world):
        """ Indices of the segments whose sampled trajectory leaves free space. """
        bad_segs = set()
        for i in range(len(self.t_waypoints) - 1):
            # Uniform samples across the segment.
            t0, t1 = self.t_waypoints[i], self.t_waypoints[i+1]
            n_samples = max(3, int(np.ceil((t1 - t0) / self.check_dt)))
            seg_pts = np.array([self._evaluate(t, 1)[0]
                                for t in np.linspace(t0, t1, n_samples)])
            if len(world.path_collisions(seg_pts, self.margin)) > 0:
                bad_segs.add(i)
        return bad_segs


    def _build(self, points):
        """ Waypoint arrival times and polynomial coefficients. """
        # Initial segment times.
        T = np.maximum(np.linalg.norm(np.diff(points, axis=0), axis=1)
                       / self.v_nom, self.dt_min)

        # Time ratio optimization and global scaling.
        T = self._optimize_ratios(points, T)
        coeffs = self._min_snap(points, T)
        T = T * self._global_scale(coeffs, T)
        return np.concatenate(([0.0], np.cumsum(T))), coeffs


    def _snap_cost(self, points, T):
        """ Integrated squared snap of the optimal trajectory for these times. """
        coeffs = self._min_snap(points, T)
        return float(sum(np.sum(coeffs[i] * (self.Q_snap @ coeffs[i])) / T[i]**7
                         for i in range(len(T))))


    def _optimize_ratios(self, points, T):
        """ Descend on the snap cost in the subspace that preserves total time. """
        n_segs = len(T)
        if n_segs < 2:
            return T
        t_total = T.sum()
        t_slack = t_total - self.dt_min * n_segs
        cost = self._snap_cost(points, T)
        step = 0.25 * t_total / n_segs

        for _ in range(self.n_time_iters):
            # Finite-difference gradient.
            h_probe = 1e-3 * t_total / n_segs
            grad = np.empty(n_segs)
            for i in range(n_segs):
                T_probe = T - h_probe / (n_segs - 1)
                T_probe[i] = T[i] + h_probe
                grad[i] = (self._snap_cost(points, np.maximum(T_probe, self.dt_min))
                           - cost) / h_probe
            grad -= grad.mean()
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-9:
                break

            # Backtracking line search.
            for _ in range(12):
                T_try = np.maximum(T - step * grad / grad_norm, self.dt_min)
                T_try_slack = T_try - self.dt_min
                T_try = self.dt_min + T_try_slack * (t_slack / T_try_slack.sum())
                cost_try = self._snap_cost(points, T_try)
                if cost_try < cost:
                    T, cost = T_try, cost_try
                    step *= 1.4
                    break
                step *= 0.5
            else:
                break
        return T


    def _global_scale(self, coeffs, T):
        """ Factor on every segment time that keeps the rotors inside their range. """
        # Derivatives 0..4 sampled along every segment.
        taus = np.linspace(0.0, 1.0, 60)
        basis_rows = [np.array([self._basis(t, order) for t in taus]) for order in range(5)]
        derivs = [np.vstack([basis_row @ coeffs[i] for i in range(len(T))])
                  for basis_row in basis_rows]
        T_rep = np.repeat(T, len(taus))[:, None]

        def exceeds_limits(scale):
            if np.linalg.norm(derivs[1] / (T_rep * scale), axis=1).max() > self.v_max:
                return True
            f_thrust, moment = self._flat_inputs(derivs[2] / (T_rep * scale)**2,
                                                 derivs[3] / (T_rep * scale)**3,
                                                 derivs[4] / (T_rep * scale)**4)
            f_rotor = np.einsum('rc,nc->nr', self.alloc_inv,
                                np.column_stack([f_thrust, moment]))
            return bool(f_rotor.max() > self.f_rotor_max or f_rotor.min() < 0.0)

        # Log-space bisection.
        scale_lo, scale_hi = 1e-3, 1e3
        if not exceeds_limits(scale_lo):
            return scale_lo
        for _ in range(60):
            scale_mid = np.sqrt(scale_lo * scale_hi)
            if exceeds_limits(scale_mid):
                scale_lo = scale_mid
            else:
                scale_hi = scale_mid
        return scale_hi


    def _flat_inputs(self, x_ddot, x_dddot, x_ddddot):
        """ Collective thrust and body moment the flat outputs demand. """
        # Desired force and body axes.
        F_des = self.mass * (x_ddot + np.array([0.0, 0.0, self.g]))
        F_des_norm = np.maximum(np.linalg.norm(F_des, axis=1), 1e-9)
        b_3_des = F_des / F_des_norm[:, None]
        a_yaw = np.array([1.0, 0.0, 0.0])
        b_2_raw = np.cross(b_3_des, a_yaw)
        b_2_raw_norm = np.maximum(np.linalg.norm(b_2_raw, axis=1), 1e-9)
        b_2_des = b_2_raw / b_2_raw_norm[:, None]
        b_1_des = np.cross(b_2_des, b_3_des)

        # Roll and pitch rates.
        F_des_dot, F_des_ddot = self.mass * x_dddot, self.mass * x_ddddot
        F_des_norm_dot = np.sum(F_des_dot * b_3_des, axis=1)
        w_x = -np.sum(F_des_dot * b_2_des, axis=1) / F_des_norm
        w_y = np.sum(F_des_dot * b_1_des, axis=1) / F_des_norm

        # Yaw rate.
        b_3_des_dot = (F_des_dot - F_des_norm_dot[:, None] * b_3_des) / F_des_norm[:, None]
        b_2_raw_dot = np.cross(b_3_des_dot, a_yaw)
        b_2_des_dot = ((b_2_raw_dot - np.sum(b_2_raw_dot * b_2_des, axis=1)[:, None] * b_2_des)
                       / b_2_raw_norm[:, None])
        b_1_des_dot = np.cross(b_2_des_dot, b_3_des) + np.cross(b_2_des, b_3_des_dot)
        w_z = np.sum(b_2_des * b_1_des_dot, axis=1)

        # Roll and pitch angular accelerations.
        w_y_dot = (np.sum(F_des_ddot * b_1_des, axis=1)
                   - 2 * F_des_norm_dot * w_y - F_des_norm * w_z * w_x) / F_des_norm
        w_x_dot = (-np.sum(F_des_ddot * b_2_des, axis=1)
                   - 2 * F_des_norm_dot * w_x + F_des_norm * w_z * w_y) / F_des_norm

        # Body moment; yaw angular acceleration dropped.
        w_des = np.column_stack([w_x, w_y, w_z])
        w_des_dot = np.column_stack([w_x_dot, w_y_dot, np.zeros_like(w_x_dot)])
        inertia_diag = np.array(self.inertia)
        return F_des_norm, w_des_dot * inertia_diag + np.cross(w_des, w_des * inertia_diag)


    @staticmethod
    def _snap_gram(n_coef):
        """ Integral of the squared 4th derivative over normalised time in [0,1]. """
        Q = np.zeros((n_coef, n_coef))
        for j in range(4, n_coef):
            for k in range(4, n_coef):
                Q[j, k] = (factorial(j) / factorial(j - 4)
                           * factorial(k) / factorial(k - 4) / (j + k - 7))
        return Q


    def _basis(self, tau, order):
        """ Row of d^order/dtau^order for the monomial basis at tau. """
        basis_row = np.zeros(self.n_coef)
        for k in range(order, self.n_coef):
            mult = 1.0
            for i in range(order):
                mult *= (k - i)
            basis_row[k] = mult * tau ** (k - order)
        return basis_row


    def _min_snap(self, points, T):
        """ Minimum-snap trajectory as a square linear solve. """
        # Constraint system.
        n_segs = len(T)
        n_coef = self.n_coef
        n_unknowns = n_coef * n_segs
        A = np.zeros((n_unknowns, n_unknowns))
        B = np.zeros((n_unknowns, 3))
        row = 0

        # Waypoint constraints.
        for i in range(n_segs):
            A[row, n_coef*i:n_coef*(i+1)] = self.basis_0[0]
            B[row] = points[i]
            row += 1
            A[row, n_coef*i:n_coef*(i+1)] = self.basis_1[0]
            B[row] = points[i+1]
            row += 1

        # Endpoint rest constraints.
        for order in range(1, 4):
            A[row, 0:n_coef] = self.basis_0[order] / T[0]**order
            row += 1
            A[row, n_coef*(n_segs-1):n_coef*n_segs] = self.basis_1[order] / T[-1]**order
            row += 1

        # Continuity of derivatives 1..6 across interior knots.
        for i in range(1, n_segs):
            for order in range(1, 7):
                A[row, n_coef*(i-1):n_coef*i] = self.basis_1[order] / T[i-1]**order
                A[row, n_coef*i:n_coef*(i+1)] = -self.basis_0[order] / T[i]**order
                row += 1

        return np.linalg.solve(A, B).reshape(n_segs, n_coef, 3)


    def _evaluate(self, t, n_orders=5):
        """ Position and the first n_orders-1 derivatives at time t. """
        # Hold start/end outside the trajectory window.
        if t <= self.t_waypoints[0]:
            return self.points[0], *(np.zeros((n_orders - 1, 3)))
        if t >= self.t_waypoints[-1]:
            return self.points[-1], *(np.zeros((n_orders - 1, 3)))

        # Segment index and normalized time.
        idx = np.searchsorted(self.t_waypoints, t, side='right') - 1
        idx = max(0, min(self.coeffs.shape[0] - 1, idx))
        T_seg = self.t_waypoints[idx+1] - self.t_waypoints[idx]
        tau = (t - self.t_waypoints[idx]) / T_seg
        c = self.coeffs[idx]

        return tuple((self._basis(tau, order) @ c) / T_seg**order for order in range(n_orders))


    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x, x_dot, x_ddot, x_dddot, x_ddddot = self._evaluate(t)

        flat_output = {
            'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot,
            'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
            'yaw': 0.0, 'yaw_dot': 0.0,
        }
        return flat_output
