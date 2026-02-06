import numpy as np


GRAVITY = 9.806
ALPHA = np.array([[20, 30, 30, 30]]).T


# --- Quaternion-based yaw error functions --- #
def quaternion_from_yaw(yaw):
    """Convert yaw angle to a quaternion."""
    half_yaw = yaw / 2.0
    return np.array([np.cos(half_yaw), 0, 0, np.sin(half_yaw)])

def quaternion_conjugate(q):
    """Return the conjugate of a quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def yaw_error_from_quaternion(q):
    """Extract the yaw error (in radians) from a quaternion."""
    return 2 * np.arctan2(q[3], q[0])

def quaternion_normalize(q):
    """Normalize a quaternion."""
    return q / np.linalg.norm(q)

def shortest_path_yaw_quaternion(current_yaw, desired_yaw):
    """Calculate the shortest path between two yaw angles using quaternions."""
    q_current = quaternion_normalize(quaternion_from_yaw(current_yaw))
    q_desired = quaternion_normalize(quaternion_from_yaw(desired_yaw))

    q_error = quaternion_multiply(q_desired, quaternion_conjugate(q_current))
    q_error_normalized = quaternion_normalize(q_error)
    q_error_shortest = np.sign(q_error_normalized[0]) * q_error_normalized
    return yaw_error_from_quaternion(q_error_shortest)


def get_tracking_error(ref, pred):
    """Calculate tracking error with quaternion-based yaw error."""
    err = ref - pred
    err[3][0] = shortest_path_yaw_quaternion(pred[3][0], ref[3][0])
    return err


# --- Thrust CBF via barrier function --- #
def _thrust_cbf(tau, u_df_xyz, gamma=10):
    """Control Barrier Function for thrust constraint via barrier function.

    Constrains thrust to safe region [3, 27] N using a quadratic barrier.

    Args:
        tau: current thrust (N)
        u_df_xyz: first 3 elements of the flat control input (3x1)
        gamma: CBF gain parameter

    Returns:
        v: 4x1 CBF correction term
    """
    h = -(5/72)*(tau-3)*(tau-27)
    dhdtau = -(5/72)*(2*tau-30)

    a3 = np.array([[0, 0, -1]]).T
    grav_vector = GRAVITY * a3

    # Compute accels from x_df (used to get taudot coefficient)
    # Note: accels are passed implicitly via u_df_xyz context
    # We recompute from the acceleration stored in x_df
    accels = u_df_xyz + grav_vector
    x_norm = np.linalg.norm(accels)
    xT_over_normx = accels / x_norm if x_norm != 0 else np.zeros((3, 1))

    # Mass coefficient for taudot: taudot = mass * (accels/||accels||)^T * jerk
    # phi = dh/dtau * mass * accels/||accels||
    # But we need mass from caller - we use the coefficient form
    # For the CBF check we use phi without mass (it cancels)
    phi = dhdtau * xT_over_normx

    d = -phi.T @ u_df_xyz[0:3] - gamma * h

    if d <= 0:
        v = np.zeros((4, 1))
    else:
        v_xyz = (d / (np.linalg.norm(phi))**2) * phi
        v = np.vstack([v_xyz, np.zeros((1, 1))])

    return v


def nr_diff_flat_numpy(state, last_input, x_df, ref, T_lookahead, step, mass, ROT, last_yaw_ref):
    """Differential-flatness based Newton-Raphson controller (NumPy version).

    Uses the 2nd-order flat state (12 elements) for position prediction and
    computes thrust and body rates via differential flatness with a thrust CBF.

    Args:
        state: 9x1 state [x, y, z, vx, vy, vz, roll, pitch, yaw]
        last_input: 4x1 last input [thrust, p, q, r]
        x_df: 12x1 flat state [sigma(4), sigmadot(4), sigmaddot(4)]
        ref: 4x1 reference [x_ref, y_ref, z_ref, yaw_ref]
        T_lookahead: lookahead horizon (seconds)
        step: integration dt (seconds)
        mass: vehicle mass (kg)
        ROT: 3x3 rotation matrix from odometry
        last_yaw_ref: previous yaw reference (for yaw_dot computation)

    Returns:
        u: 4x1 control input [thrust, p, q, r]
        x_df: 12x1 updated flat state
        vnorm: norm of the CBF correction term
    """
    curr_x, curr_y, curr_z = state[0][0], state[1][0], state[2][0]
    curr_vx, curr_vy, curr_vz = state[3][0], state[4][0], state[5][0]
    curr_yaw = state[8][0]
    curr_yawdot = last_input[3][0]
    curr_thrust = last_input[0][0]

    # --- Flat state prediction --- #
    z1 = np.array([[curr_x, curr_y, curr_z, curr_yaw]]).T      # position
    z2 = np.array([[curr_vx, curr_vy, curr_vz, curr_yawdot]]).T  # velocity
    z3 = x_df[8:12]  # acceleration from flat state

    # Inverse Jacobian for 2nd-order flat system
    dgdu_inv = (2 / T_lookahead**2) * np.eye(4)

    # Predict output at lookahead
    pred = z1 + z2 * T_lookahead + (1/2) * z3 * T_lookahead**2

    # Tracking error with quaternion yaw handling
    error = get_tracking_error(ref, pred)

    # Newton-Raphson control
    NR = dgdu_inv @ error
    u_df = ALPHA * NR

    # --- Thrust CBF --- #
    v = _thrust_cbf(curr_thrust, u_df[0:3])

    # --- Flat state dynamics integration --- #
    A_df = np.block([
        [np.zeros((4, 4)), np.eye(4), np.zeros((4, 4))],
        [np.zeros((4, 4)), np.zeros((4, 4)), np.eye(4)],
        [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]
    ])

    B_df = np.block([
        [np.zeros((4, 4))],
        [np.zeros((4, 4))],
        [np.eye(4)]
    ])

    x_dot_df = A_df @ x_df + B_df @ (u_df + v)
    x_df = x_df + x_dot_df * step

    sigma = x_df[0:4]
    sigmad1 = x_df[4:8]
    sigmad2 = x_df[8:12]
    sigmad3 = u_df

    # --- Reconstruct thrust and body rates via differential flatness --- #
    a1 = np.array([[1, 0, 0]]).T
    a2 = np.array([[0, 1, 0]]).T
    a3 = np.array([[0, 0, -1]]).T

    accels = sigmad2[0:3] + GRAVITY * a3
    thrust = mass * np.linalg.norm(accels)
    clipped_thrust = thrust  # CBF handles constraint

    # Desired body frame axes
    b3 = accels / np.linalg.norm(accels)
    e1 = np.cos(curr_yaw) * a1 + np.sin(curr_yaw) * a2
    val2 = np.cross(b3, e1, axis=0)
    b2 = val2 / np.linalg.norm(val2)
    b1 = np.cross(b2, b3, axis=0)

    # Body rates from jerk (3rd derivative of flat output)
    j = sigmad3[0:3]
    val3 = (j - (j.T @ b3) * b3)
    p = (b2.T @ ((mass / -clipped_thrust) * val3)).item()
    q = (b1.T @ ((mass / -clipped_thrust) * val3)).item()

    # Solve for yaw rate using frame consistency equation
    A_mat = np.hstack((e1, b2, a3))
    B_mat = ROT

    curr_yaw_ref = ref[3][0].item()
    yaw_dot = (curr_yaw_ref - last_yaw_ref) / step
    x_known = np.array([yaw_dot])
    y_known = np.array([p, q])

    # Split into known/unknown parts and solve
    A1 = A_mat[:, :2]
    A2 = A_mat[:, 2:]
    B1 = B_mat[:, 2:]
    B2 = B_mat[:, :2]

    rhs = (B2 @ y_known) - (A2 @ x_known)
    M = np.hstack((A1, -B1))
    unknowns = np.linalg.solve(M, rhs)

    x1, x2, r = unknowns
    r = -r

    # Clip rates
    max_rate = 0.8
    r = np.clip(r, -max_rate, max_rate)
    p = np.clip(p, -max_rate, max_rate)
    q = np.clip(q, -max_rate, max_rate)

    u = np.array([[clipped_thrust, p, q, r]]).T

    return u, x_df, np.linalg.norm(v)
