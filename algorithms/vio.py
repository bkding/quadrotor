import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple.
    p, v, q, a_b, w_b, g = nominal_state

    # STUDENT CODE HERE
    R = q.as_matrix()

    # Position, velocity, and orientation propagation.
    new_p = p + v * dt + 0.5 * (R @ (a_m - a_b) + g) * dt**2
    new_v = v + (R @ (a_m - a_b) + g) * dt
    new_q = q * Rotation.from_rotvec(((w_m - w_b) * dt).ravel())

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """
    # Unpack nominal_state tuple.
    p, v, q, a_b, w_b, g = nominal_state

    # STUDENT CODE HERE
    R   = q.as_matrix()
    I3  = np.eye(3)

    # Skew-symmetric matrix of (a_m - a_b).
    a = (a_m - a_b).ravel()
    a_skew = np.array([[    0, -a[2],  a[1]],
                       [ a[2],     0, -a[0]],
                       [-a[1],  a[0],     0]])

    # Rotation matrix from rotation vector (w_m - w_b)*dt.
    R_wdt = Rotation.from_rotvec(((w_m - w_b) * dt).ravel()).as_matrix()

    # Error state transition matrix Fx (18x18).
    Fx = np.eye(18)
    Fx[0:3, 3:6]   = I3 * dt                       # dp/dv
    Fx[3:6, 6:9]   = -R @ a_skew * dt              # dv/dtheta
    Fx[3:6, 9:12]  = -R * dt                       # dv/dab
    Fx[3:6, 15:18] = I3 * dt                       # dv/dg
    Fx[6:9, 6:9]   = R_wdt.T                       # dtheta/dtheta
    Fx[6:9, 12:15] = -I3 * dt                      # dtheta/dwb

    # Noise input matrix Fi (18x12).
    Fi = np.zeros((18, 12))
    Fi[3:6, 0:3]    = I3       # velocity noise (accelerometer)
    Fi[6:9, 3:6]    = I3       # orientation noise (gyroscope)
    Fi[9:12, 6:9]   = I3       # accel bias random walk
    Fi[12:15, 9:12] = I3       # gyro bias random walk

    # Continuous noise covariance Qi (12x12).
    Qi = np.zeros((12, 12))
    Qi[0:3, 0:3]   = accelerometer_noise_density**2 * dt**2 * I3
    Qi[3:6, 3:6]   = gyroscope_noise_density**2 * dt**2 * I3
    Qi[6:9, 6:9]   = accelerometer_random_walk**2 * dt * I3
    Qi[9:12, 9:12] = gyroscope_random_walk**2 * dt * I3

    # Covariance propagation.
    P = Fx @ error_state_covariance @ Fx.T + Fi @ Qi @ Fi.T

    return P


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    # Unpack nominal_state tuple.
    p, v, q, a_b, w_b, g = nominal_state

    # STUDENT CODE HERE
    R = q.as_matrix()

    # Transform world point to camera frame.
    Pc = R.T @ (Pw - p)
    Xc, Yc, Zc = Pc.ravel()

    # Predicted measurement.
    uv_hat = np.array([[Xc / Zc],
                       [Yc / Zc]])

    # Innovation.
    innovation = uv - uv_hat

    # Outlier rejection.
    if norm(innovation) > error_threshold:
        return nominal_state, error_state_covariance, innovation

    # Measurement Jacobian w.r.t. camera point (2x3).
    dz_dPc = (1.0 / Zc) * np.array([[1, 0, -Xc / Zc],
                                     [0, 1, -Yc / Zc]])

    # Skew-symmetric matrix of Pc.
    Pc_skew = np.array([[     0, -Zc,  Yc],
                        [   Zc,    0, -Xc],
                        [  -Yc,  Xc,    0]])

    # Observation matrix Ht (2x18).
    Ht = np.zeros((2, 18))
    Ht[:, 0:3] = dz_dPc @ (-R.T)          # w.r.t. δp
    Ht[:, 6:9] = dz_dPc @ Pc_skew         # w.r.t. δθ

    # Kalman gain.
    S  = Ht @ error_state_covariance @ Ht.T + Q
    Kt = error_state_covariance @ Ht.T @ inv(S)

    # Error state correction.
    dx = Kt @ innovation

    # Update nominal state.
    new_p   = p + dx[0:3]
    new_v   = v + dx[3:6]
    new_q   = q * Rotation.from_rotvec(dx[6:9].ravel())
    new_a_b = a_b + dx[9:12]
    new_w_b = w_b + dx[12:15]
    new_g   = g + dx[15:18]

    # Update covariance (Joseph form).
    I_KH = np.eye(18) - Kt @ Ht
    new_error_state_covariance = I_KH @ error_state_covariance @ I_KH.T + Kt @ Q @ Kt.T

    return (new_p, new_v, new_q, new_a_b, new_w_b, new_g), new_error_state_covariance, innovation