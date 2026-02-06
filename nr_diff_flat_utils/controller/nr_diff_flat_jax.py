import jax.numpy as jnp
from nr_diff_flat_utils.jax_utils import jit

GRAVITY = 9.806
ALPHA = jnp.array([[20, 30, 30, 30]]).T

@jit
def NR_tracker_flat(STATE, INPUT, x_df, ref, T_LOOKAHEAD, step, MASS, ROT, yaw_dot):
    """Differential-flatness based Newton-Raphson controller (JAX version).

    Uses the flat output [x, y, z, yaw] and its derivatives to compute
    thrust and body rates directly via differential geometry.

    Args:
        STATE: 9x1 state [x, y, z, vx, vy, vz, roll, pitch, yaw]
        INPUT: 4x1 last input [thrust, p, q, r]
        x_df: 12x1 flat state [sigma(4), sigmadot(4), sigmaddot(4)]
        ref: 4x1 reference [x_ref, y_ref, z_ref, yaw_ref]
        T_LOOKAHEAD: lookahead horizon (seconds)
        step: integration dt (seconds)
        MASS: vehicle mass (kg)
        ROT: 3x3 rotation matrix from odometry
        yaw_dot: yaw reference rate (rad/s)

    Returns:
        u: 4x1 control input [thrust, p, q, r]
        x_df: 12x1 updated flat state
    """
    curr_x, curr_y, curr_z, curr_yaw = STATE[0][0], STATE[1][0], STATE[2][0], STATE[-1][0]
    curr_vx, curr_vy, curr_vz = STATE[3][0], STATE[4][0], STATE[5][0]
    curr_yawdot = INPUT[3][0]

    z1 = jnp.array([[curr_x, curr_y, curr_z, curr_yaw]]).T
    z2 = jnp.array([[curr_vx, curr_vy, curr_vz, curr_yawdot]]).T
    z3 = x_df[8:12]

    dgdu_inv = (2 / T_LOOKAHEAD**2) * jnp.eye(4)
    pred = z1 + z2 * T_LOOKAHEAD + (1 / 2) * z3 * T_LOOKAHEAD**2

    error = ref - pred
    NR = dgdu_inv @ error
    u_df = ALPHA * NR

    A_df = jnp.block([
                [jnp.zeros((4, 4)), jnp.eye(4), jnp.zeros((4, 4))],
                [jnp.zeros((4, 4)), jnp.zeros((4, 4)), jnp.eye(4)],
                [jnp.zeros((4, 4)), jnp.zeros((4, 4)), jnp.zeros((4, 4))]
            ])

    B_df = jnp.block([
                [jnp.zeros((4, 4))],
                [jnp.zeros((4, 4))],
                [jnp.eye(4)]
            ])

    x_dot_df = A_df @ x_df + B_df @ u_df
    x_df = x_df + x_dot_df * step

    sigma = x_df[0:4]
    sigmad1 = x_df[4:8]
    sigmad2 = x_df[8:12]
    sigmad3 = u_df

    a1 = jnp.array([[1,0,0]]).T
    a2 = jnp.array([[0,1,0]]).T
    a3 = jnp.array([[0, 0, -1]]).T

    accels = sigmad2[0:3] + GRAVITY * a3
    thrust = MASS * jnp.linalg.norm(accels)

    thrust_max = 27.0
    thrust_min = 0.5
    clipped_thrust = jnp.clip(thrust, thrust_min, thrust_max)

    b3 = accels / jnp.linalg.norm(accels)

    e1 = jnp.cos(curr_yaw) * a1 + jnp.sin(curr_yaw) * a2
    val2 = jnp.cross(b3, e1, axis=0)
    b2 = val2 / jnp.linalg.norm(val2)
    b1 = jnp.cross(b2, b3, axis=0)

    j = sigmad3[0:3]
    val3 = j - (j.T @ b3) * b3

    p = (b2.T @ ((MASS / -clipped_thrust) * val3))
    q = (b1.T @ ((MASS / -clipped_thrust) * val3))

    A_mat = jnp.hstack((e1, b2, a3))
    B_mat = ROT

    x_known = jnp.array([yaw_dot])
    y_known = jnp.array([p[0][0], q[0][0]])

    # Split A into parts affecting unknowns (A1) and knowns (A2)
    A1 = A_mat[:, :2]  # First two columns (unknown x1, x2)
    A2 = A_mat[:, 2:]  # Last column (known x3)

    # Split B into parts affecting unknowns (B1) and knowns (B2)
    B1 = B_mat[:, 2:]  # Last column (unknown y3)
    B2 = B_mat[:, :2]  # First two columns (known y1, y2)

    # Compute right-hand side
    rhs = (B2 @ y_known) - (A2 @ x_known)  # Move known values to RHS

    # Solve system for [x1, x2, y3]
    M = jnp.hstack((A1, -B1))  # Construct coefficient matrix
    unknowns = jnp.linalg.solve(M, rhs)  # Solve for unknowns

    # Extract solutions
    x1, x2, r = unknowns
    r = -r

    max_rate = 0.8
    p = jnp.clip(p, -max_rate, max_rate)
    q = jnp.clip(q, -max_rate, max_rate)
    r = jnp.clip(r, -max_rate, max_rate)

    u = jnp.array([clipped_thrust.reshape((1,1)), p.reshape((1,1)), q.reshape((1,1)), r.reshape((1,1))]).reshape((4,1))
    return u, x_df
