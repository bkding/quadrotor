import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py
        """
        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        # Control gains.
        self.K_p = np.diag([4.8, 4.8, 9.6])
        self.K_d = np.diag([3.0, 3.0, 6.0])
        self.K_R = np.diag([460.0, 460.0, 48.0])
        self.K_w = np.diag([29.0, 29.0, 26.0])

        # Desired angular velocity cap.
        self.w_des_max = 8.0 # rad/s


    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros(4)
        cmd_thrust = 0
        cmd_moment = np.zeros(3)
        cmd_q = np.zeros(4)

        # STUDENT CODE HERE
        # Current state.
        x = state['x']
        v = state['v']
        q = state['q']
        w = state['w']

        # Desired flat outputs.
        x_des = flat_output['x']
        x_dot_des = flat_output['x_dot']
        x_ddot_des = flat_output['x_ddot']
        x_dddot_des = flat_output['x_dddot']
        yaw_des = flat_output['yaw']
        yaw_dot_des = flat_output['yaw_dot']

        # Desired translational acceleration command.
        x_ddot_cmd = (x_ddot_des
                      - self.K_d @ (v - x_dot_des)
                      - self.K_p @ (x - x_des))

        # Desired force.
        F_des = self.mass * x_ddot_cmd + np.array([0.0, 0.0, self.mass * self.g]).T

        # Current attitude and thrust.
        R = Rotation.from_quat(q).as_matrix()
        b_3 = R @ np.array([0.0, 0.0, 1.0]).T
        u_1 = b_3.T @ F_des

        # Desired body z-axis.
        F_des_norm = np.linalg.norm(F_des)
        if F_des_norm < 1e-6:
            b_3_des = np.array([0.0, 0.0, 1.0]).T
        else:
            b_3_des = F_des / F_des_norm

        # Desired yaw direction.
        a_yaw = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0]).T

        # Desired body y-axis.
        b_2_raw = np.cross(b_3_des, a_yaw)
        b_2_raw_norm = np.linalg.norm(b_2_raw)
        if b_2_raw_norm < 1e-6:
            b_2_fallback = np.cross(b_3_des, np.array([1.0, 0.0, 0.0]).T)
            b_2_des = b_2_fallback / np.linalg.norm(b_2_fallback)
        else:
            b_2_des = b_2_raw / b_2_raw_norm

        # Desired rotation matrix.
        b_1_des = np.cross(b_2_des, b_3_des)
        R_des = np.column_stack((b_1_des, b_2_des, b_3_des))

        # Desired angular velocity from jerk and yaw rate.
        if b_2_raw_norm >= 1e-6 and F_des_norm >= 1e-6:
            F_des_dot = self.mass * x_dddot_des
            b_3_des_dot = (F_des_dot - (b_3_des @ F_des_dot) * b_3_des) / F_des_norm

            a_yaw_dot = yaw_dot_des * np.array([-np.sin(yaw_des), np.cos(yaw_des), 0.0]).T
            b_2_raw_dot = np.cross(b_3_des_dot, a_yaw) + np.cross(b_3_des, a_yaw_dot)
            b_2_des_dot = (b_2_raw_dot - (b_2_des @ b_2_raw_dot) * b_2_des) / b_2_raw_norm

            w_des = np.array([-b_3_des_dot @ b_2_des,
                              b_3_des_dot @ b_1_des,
                              b_2_des @ np.cross(b_2_des_dot, b_3_des)])
            w_des = np.clip(w_des, -self.w_des_max, self.w_des_max)
        else:
            w_des = np.zeros(3)

        # Attitude and angular velocity errors.
        e_R_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([e_R_matrix[2, 1],
                        e_R_matrix[0, 2],
                        e_R_matrix[1, 0]])

        e_w = w - R.T @ R_des @ w_des
        u_2 = self.inertia @ (-self.K_R @ e_R - self.K_w @ e_w)

        # Rotor force allocation.
        l = self.arm_length
        gamma = self.k_drag / self.k_thrust

        A = np.array([[1, 1, 1, 1],
                      [0, l, 0, -l],
                      [-l, 0, l, 0],
                      [gamma, -gamma, gamma, -gamma]])
        U = np.array([u_1, u_2[0], u_2[1], u_2[2]])
        F = np.linalg.solve(A, U)

        # Motor speeds.
        omega_sq = F / self.k_thrust
        omega_sq = np.clip(omega_sq, 0, None)
        motor_speeds = np.sqrt(omega_sq)

        cmd_motor_speeds = np.clip(motor_speeds,
                                   self.rotor_speed_min,
                                   self.rotor_speed_max)
        cmd_thrust = u_1
        cmd_moment = u_2
        cmd_q = Rotation.from_matrix(R_des).as_quat()

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment,
                         'cmd_q': cmd_q}
        return control_input
