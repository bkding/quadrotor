import numpy as np

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
        self.resolution    = np.array([0.25, 0.25, 0.25]) # m
        self.margin        = 0.40                         # m
        self.clearance_max = 0.30                         # m

        # Dense path and sparse waypoints.
        self.path, _ = graph_search(world, self.resolution, self.margin,
                                    start, goal, astar=True)
        self.points = self._prune_waypoints(world, self.path)
        n_pts = self.points.shape[0]

        # Per-segment profile.
        seg_vmax, seg_amax, seg_t_scale, z_ratio, cl_extra = \
            self._segment_profile(world, n_pts)

        # Corner velocities at interior waypoints.
        v_wp, speeds = self._corner_velocities(
            n_pts, seg_vmax, seg_amax, z_ratio, cl_extra)
        a_wp = np.zeros((n_pts, 3))

        # Waypoint arrival times.
        self.t_waypoints = self._waypoint_times(
            n_pts, speeds, seg_vmax, seg_amax, seg_t_scale)

        # Quintic polynomial fit.
        self.coeffs = self._fit_quintics(n_pts, v_wp, a_wp)


    def _prune_waypoints(self, world, path):
        """ Greedy line-of-sight pruning: keep farthest collision-free hop. """
        pruned = [path[0]]
        if path.shape[0] > 2:
            i = 0
            while i < path.shape[0] - 1:
                j_best = i + 1
                for j in range(path.shape[0] - 1, i, -1):
                    seg = np.vstack((path[i], path[j]))
                    cols = world.path_collisions(seg, self.margin)
                    if len(cols) == 0:
                        j_best = j
                        break
                pruned.append(path[j_best])
                i = j_best
            return np.array(pruned)
        else:
            return path


    def _segment_profile(self, world, n_pts):
        """ Per-segment geometry, kinematic limits, and time scaling. """
        # Base kinematic limits.
        v_nom   = 2.65 # m/s
        a_nom   = 8.8  # m/s^2
        v_min   = 1.65 # m/s
        v_max   = 3.35 # m/s
        a_min   = 6.6  # m/s^2
        t_scale = 0.79

        # Clearance probing buffers beyond self.margin.
        clearance_steps = (0.30, 0.20, 0.10, 0.0) # m

        # Shaping gains.
        kv_long = 0.54
        kv_z    = 0.12
        kv_turn = 0.18
        kv_cl   = 0.07
        ka_z    = 0.08
        kt_z    = 0.06
        kt_turn = 0.22
        kt_cl   = 0.08

        # Segment vectors and lengths.
        seg_vecs    = self.points[1:] - self.points[:-1]
        seg_lengths = np.linalg.norm(seg_vecs, axis=1)
        valid       = seg_lengths > 1e-4

        # Z-projection ratio and length saturation.
        z_ratio = np.zeros(n_pts - 1)
        z_ratio[valid] = np.abs(seg_vecs[valid, 2]) / seg_lengths[valid]
        long_ratio = np.clip((seg_lengths - 1.5) / 4.0, 0.0, 1.0)

        # Free-space clearance per segment.
        cl_extra = np.zeros(n_pts - 1)
        for i in range(n_pts - 1):
            seg = np.vstack((self.points[i], self.points[i+1]))
            for extra in clearance_steps:
                if len(world.path_collisions(seg, self.margin + extra)) == 0:
                    cl_extra[i] = extra
                    break
        cl_ratio = cl_extra / self.clearance_max

        # Horizontality and turn severity.
        h_ratio = 1.0 - z_ratio
        turn_sev = np.zeros(n_pts - 1)
        for i in range(1, n_pts - 1):
            if seg_lengths[i-1] > 1e-4 and seg_lengths[i] > 1e-4:
                dir_prev = seg_vecs[i-1] / seg_lengths[i-1]
                dir_next = seg_vecs[i] / seg_lengths[i]
                dot_turn = np.clip(np.dot(dir_prev, dir_next), -1.0, 1.0)
                severity = 0.5 * (1.0 - dot_turn)
                turn_sev[i-1] = max(turn_sev[i-1], severity)
                turn_sev[i] = max(turn_sev[i], severity)
        h_turn = turn_sev * h_ratio**1.2

        # Per-segment v_max.
        seg_vmax = v_nom * (1.0 + kv_long * h_ratio * long_ratio)
        seg_vmax *= (1.0 - kv_z * z_ratio**1.5)
        seg_vmax *= (1.0 - kv_turn * h_turn)
        seg_vmax *= (0.93 + kv_cl * cl_ratio)
        seg_vmax = np.clip(seg_vmax, v_min, v_max)

        # Per-segment a_max.
        seg_amax = a_nom * (1.0 - ka_z * z_ratio**1.2)
        seg_amax = np.clip(seg_amax, a_min, a_nom)

        # Per-segment time scale.
        seg_t_scale = t_scale * (1.0 + kt_z * z_ratio**1.4)
        seg_t_scale *= (1.0 + kt_turn * h_turn)
        seg_t_scale *= (1.0 + kt_cl * (1.0 - cl_ratio))

        return seg_vmax, seg_amax, seg_t_scale, z_ratio, cl_extra


    def _corner_velocities(self, n_pts, seg_vmax, seg_amax, z_ratio, cl_extra):
        """ Heuristic speeds at interior waypoints; endpoints stay at rest. """
        # Corner-speed shaping gains.
        kc_z  = 0.10
        kc_cl = 0.12

        v_wp   = np.zeros((n_pts, 3))
        speeds = np.zeros(n_pts)

        for i in range(1, n_pts - 1):
            # Adjacent segments.
            seg_prev = self.points[i] - self.points[i-1]
            seg_next = self.points[i+1] - self.points[i]
            n_prev = np.linalg.norm(seg_prev)
            n_next = np.linalg.norm(seg_next)

            if n_prev > 1e-4 and n_next > 1e-4:
                # Unit directions and turn cosine.
                dir_prev = seg_prev / n_prev
                dir_next = seg_next / n_next
                dot_prod = np.dot(dir_prev, dir_next)

                # Reachable speed from rest.
                a_turn = min(seg_amax[i-1], seg_amax[i])
                v_turn_limit = min(seg_vmax[i-1], seg_vmax[i])
                s_prev = np.sqrt(a_turn * n_prev)
                s_next = np.sqrt(a_turn * n_next)

                # Turn-angle multiplier.
                if dot_prod >= 0.9:
                    multiplier = 1.0
                elif dot_prod > 0:
                    multiplier = 0.3 + 0.7 * dot_prod
                else:
                    multiplier = max(0.05, 0.2 + 0.15 * dot_prod)

                # Verticality and clearance scaling.
                z_turn = max(z_ratio[i-1], z_ratio[i])
                cl_turn = min(cl_extra[i-1], cl_extra[i]) / self.clearance_max
                multiplier *= (1.0 - kc_z * z_turn**1.3)
                multiplier *= (0.88 + kc_cl * cl_turn)
                corner_speed = min(v_turn_limit, s_prev, s_next) * multiplier
                speeds[i] = corner_speed

                # Bisector direction.
                dir_v = (dir_prev + dir_next)
                dir_n = np.linalg.norm(dir_v)
                if dir_n > 1e-4:
                    dir_v = dir_v / dir_n
                else:
                    dir_v = dir_next

                v_wp[i] = dir_v * corner_speed

        return v_wp, speeds


    def _waypoint_times(self, n_pts, speeds, seg_vmax, seg_amax, seg_t_scale):
        """ Cumulative arrival times from trapezoidal/triangular speed profile. """
        dt_min = 0.1 # s

        t_waypoints = [0.0]
        for i in range(1, n_pts):
            # Segment endpoints and limits.
            d = np.linalg.norm(self.points[i] - self.points[i-1])
            v_start = speeds[i-1]
            v_end = speeds[i]
            v_max = seg_vmax[i-1]
            a_max = seg_amax[i-1]

            # Triangular or trapezoidal speed profile.
            v_peak = np.sqrt((v_start**2 + v_end**2)/2.0 + a_max * d)
            if v_peak < v_max:
                dt = (2*v_peak - v_start - v_end) / a_max
            else:
                d_acc = (v_max**2 - v_start**2)/(2*a_max) + (v_max**2 - v_end**2)/(2*a_max)
                t_cruise = max(0, d - d_acc) / v_max
                dt = (v_max - v_start)/a_max + (v_max - v_end)/a_max + t_cruise

            # Floor segment duration to bound peak acceleration.
            dt = dt * seg_t_scale[i-1]
            if dt < dt_min:
                dt = dt_min
            t_waypoints.append(t_waypoints[-1] + dt)

        return np.array(t_waypoints)


    def _fit_quintics(self, n_pts, v_wp, a_wp):
        """ Solve per-segment quintic polynomials matching p, v, a at endpoints. """
        coeffs = np.zeros((6, n_pts - 1, 3))
        for i in range(n_pts - 1):
            # Segment endpoints.
            T_seg = self.t_waypoints[i+1] - self.t_waypoints[i]
            p0 = self.points[i]
            p1 = self.points[i+1]
            v0 = v_wp[i]
            v1 = v_wp[i+1]
            a0 = a_wp[i]
            a1 = a_wp[i+1]

            # Lower-order coefficients.
            c0 = p0
            c1 = v0
            c2 = a0 / 2.0

            # Higher-order coefficients via 3x3 solve.
            A = np.array([
                [T_seg**3, T_seg**4, T_seg**5],
                [3*T_seg**2, 4*T_seg**3, 5*T_seg**4],
                [6*T_seg, 12*T_seg**2, 20*T_seg**3]
            ])
            B = np.array([
                p1 - (c0 + c1*T_seg + c2*T_seg**2),
                v1 - (c1 + 2*c2*T_seg),
                a1 - (2*c2)
            ])
            sol = np.linalg.solve(A, B)
            coeffs[0, i] = c0
            coeffs[1, i] = c1
            coeffs[2, i] = c2
            coeffs[3, i] = sol[0]
            coeffs[4, i] = sol[1]
            coeffs[5, i] = sol[2]

        return coeffs


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
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw      = 0.5
        yaw_dot  = 0

        # Hold start/end outside the trajectory window.
        if t <= self.t_waypoints[0]:
            x = self.points[0]
        elif t >= self.t_waypoints[-1]:
            x = self.points[-1]
        else:
            # Segment index and local time.
            idx = np.searchsorted(self.t_waypoints, t, side='right') - 1
            idx = max(0, min(self.points.shape[0] - 2, idx))
            t_seg = t - self.t_waypoints[idx]
            c = self.coeffs[:, idx, :]

            # Time powers.
            t2 = t_seg * t_seg
            t3 = t2 * t_seg
            t4 = t3 * t_seg
            t5 = t4 * t_seg

            # Polynomial basis and derivatives.
            basis = np.array([
                [1, t_seg, t2,    t3,      t4,       t5],
                [0, 1,     2*t_seg, 3*t2,  4*t3,     5*t4],
                [0, 0,     2,     6*t_seg, 12*t2,    20*t3],
                [0, 0,     0,     6,       24*t_seg, 60*t2],
                [0, 0,     0,     0,       24,       120*t_seg],
            ])
            x, x_dot, x_ddot, x_dddot, x_ddddot = basis @ c

        flat_output = {
            'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot,
            'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
            'yaw': yaw, 'yaw_dot': yaw_dot,
        }
        return flat_output