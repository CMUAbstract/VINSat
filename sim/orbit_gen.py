from scipy.spatial import transform
import numpy as np

class OrbitalElements:
    def __init__(self, a, e, i, Omega, omega, nu):
        self.a = a
        self.e = e
        self.i = i
        self.Omega = Omega
        self.omega = omega
        self.nu = nu

def oe2eci(oe, mu=398600.4418):
    P = oe.a * (1 - oe.e**2)
    r_mag = P / (1 + oe.e * np.cos(oe.nu))
    n = np.sqrt(mu / oe.a**3)
    E = anom2E(oe.nu, oe.e)

    r_peri = np.array([oe.a * (np.cos(E) - oe.e), oe.a * np.sqrt(1 - oe.e**2) * np.sin(E), 0])
    v_periComp = np.array([-np.sin(E), np.sqrt(1 - oe.e**2) * np.cos(E), 0])
    v_peri = (oe.a * n) / (1 - oe.e * np.cos(E)) * v_periComp

    if oe.i == 0 and oe.e != 0:
        R1 = np.eye(3)
        R2 = np.eye(3)
        R3 = rotz(oe.omega)
    elif oe.e == 0 and oe.i != 0:
        R1 = rotz(oe.Omega)
        R2 = rotx(oe.i)
        R3 = np.eye(3)
    elif oe.i == 0 and oe.e == 0:
        R1 = np.eye(3)
        R2 = np.eye(3)
        R3 = np.eye(3)
    else:
        R1 = rotz(oe.Omega)
        R2 = rotx(oe.i)
        R3 = rotz(oe.omega)

    R = np.dot(R1, np.dot(R2, R3))
    r_eci = np.dot(R, r_peri)
    v_eci = np.dot(R, v_peri)

    return np.concatenate([r_eci, v_eci])

def anom2E(nu, e):
    E = np.arccos((e + np.cos(nu)) / (1 + e * np.cos(nu)))
    if nu > np.pi:
        E = 2 * np.pi - E
    return E

def rotz(gamma):
    return np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma),  np.cos(gamma), 0],
        [0,              0,             1]
    ])

def rotx(alpha):
    return np.array([
        [1,          0,              0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)]
    ])


def eci2oe(x, mu=398600.4418):
    R = x[:3]
    V = x[3:6]

    r = np.linalg.norm(R)
    v = np.linalg.norm(V)

    H = np.cross(R, V)
    h = np.linalg.norm(H)

    N = np.cross([0, 0, 1], H)
    n = np.linalg.norm(N)

    e_vec = (1 / mu) * ((v**2 - mu / r) * R - np.dot(R, V) * V)
    e = np.linalg.norm(e_vec)

    epsilon = 0.5 * v**2 - mu / r
    if e != 1:
        a = -mu / (2 * epsilon)
    else:
        a = float('inf')

    i = np.arccos(H[2] / h)

    Omega = np.arccos(N[0] / n)
    if N[1] < 0:
        Omega = 2 * np.pi - Omega

    term = np.dot(N, e_vec) / (n * e)
    epsilon_val = 1e-10
    if abs(term) > 1 and abs(term) - 1 < epsilon_val:
        term = np.sign(term)

    omega = np.arccos(term)
    if e_vec[2] < 0:
        omega = 2 * np.pi - omega

    term = np.dot(e_vec, R) / (e * r)
    if abs(term) > 1 and abs(term) - 1 < epsilon_val:
        term = np.sign(term)

    nu = np.arccos(term)
    if np.dot(R, V) < 0:
        nu = 2 * np.pi - nu

    if i == 0 and e != 0:
        ang = np.arccos(e_vec[0] / e)
        if e_vec[1] < 0:
            ang = 2 * np.pi - ang
    elif i != 0 and e == 0:
        ang = np.arccos(np.dot(N, R) / (n * r))
        if R[2] < 0:
            ang = 2 * np.pi - ang
    elif i == 0 and e == 0:
        ang = np.arccos(R[0] / r)
        if R[1] < 0:
            ang = 2 * np.pi - ang
    else:
        ang = np.nan

    oe = OrbitalElements(a, e, i, Omega, omega, nu)
    return oe, ang

def orbit_dynamics(x_orbit, mu=398600.4418, J2=1.75553e10):
    r = x_orbit[:3]
    v = x_orbit[3:6]

    r_mat = np.array([
        [6, -1.5, -1.5],
        [6, -1.5, -1.5],
        [3, -4.5, -4.5]
    ])

    # v_dot = -(mu / np.linalg.norm(r)**3) * r + (J2 / np.linalg.norm(r)**7) * np.dot(r, r_mat)
    v_dot = -(mu / np.linalg.norm(r)**3) * r + (J2 / np.linalg.norm(r)**7) * np.dot(r_mat, r**2) * r

    return np.concatenate([v, v_dot])

def orbit_step(xk, h):
    f1 = orbit_dynamics(xk)
    f2 = orbit_dynamics(xk + 0.5 * h * f1)
    f3 = orbit_dynamics(xk + 0.5 * h * f2)
    f4 = orbit_dynamics(xk + h * f3)

    xn = xk + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    return xn


# 3U CubeSat Inertia (MKS units)
m = 4.0
J = np.diag([(m/12) * (.1**2 + .34**2), (m/12) * (.1**2 + .34**2), (m/12) * (.1**2 + .1**2)])

def hat(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def L(q):
    s = q[0]
    v = q[1:]
    return np.concatenate([np.concatenate([[s], -v.T])[None], np.concatenate([v[:,None], s * np.eye(3) + hat(v)], axis=1)], axis=0)

T = np.diag([1, -1, -1, -1])
H = np.concatenate([np.zeros((1,3)), np.eye(3)], axis=0)

def qtoQ(q):
    return H.T @ T @ L(q) @ T @ L(q) @ H

def G(q):
    return L(q) @ H

def rptoq(phi):
    return (1/np.sqrt(1 + phi.T @ phi)) * np.hstack((1, phi))

def qtorp(q):
    return q[1:] / q[0]

def attitude_dynamics(x_attitude, J=np.diag((1/3) * np.array([(.1**2 + .34**2), (.1**2 + .34**2), (.1**2 + .1**2)]))):
    q = x_attitude[:4]
    q /= np.linalg.norm(q)
    omega = x_attitude[4:]

    q_dot = 0.5 * G(q) @ omega
    ### switch with scipy solve ###
    # omega_dot = -np.linalg.inv(J) @ (hat(omega) @ J @ omega)
    omega_dot = -np.linalg.solve(J, (hat(omega) @ J @ omega))


    return np.hstack((q_dot, omega_dot))

def attitude_step(xk, h):
    f1 = attitude_dynamics(xk)
    f2 = attitude_dynamics(xk + 0.5 * h * f1)
    f3 = attitude_dynamics(xk + 0.5 * h * f2)
    f4 = attitude_dynamics(xk + h * f3)

    xn = xk + (h/6.0) * (f1 + 2*f2 + 2*f3 + f4)
    xn[:4] /= np.linalg.norm(xn[:4])  # re-normalize quaternion

    return xn

def generate_eci_traj(oe, tf=3*60*60, ts=1, no_velocities=True):

    # Get initial ECI position and velocity of orbit
    x0_orbit = oe2eci(oe)
    
    # Simulate for given duration with given sample rate
    tsamp = np.arange(0, tf+1, ts)
    
    # Simulate position 
    xtraj_orbit = np.zeros((6, len(tsamp)))
    xtraj_orbit[:, 0] = x0_orbit
    
    for k in range(len(tsamp) - 1):
        xtraj_orbit[:, k+1] = orbit_step(xtraj_orbit[:, k], ts)
        
    # Simulate attitude
    # Initial attitude conditions
    ### OLD CODE ###
    q0 = np.ones(4)*np.random.randn(4)
    q0 /= np.linalg.norm(q0)
    omega0 = 2 * (np.pi/180) * np.ones(3)*np.random.randn(3)
    x0_attitude = np.hstack((q0, omega0))

    xtraj_attitude = np.zeros((7, len(tsamp)))
    xtraj_attitude[:, 0] = x0_attitude

    for k in range(len(tsamp)-1):
        xtraj_attitude[:, k+1] = attitude_step(xtraj_attitude[:, k], 1.0)
            
    ### NEW UNWORKING CODE ###
    #xtraj_attitude = convert_pos_to_quaternion(xtraj_orbit[:,:3])
    
    dir_vec, up_vec, right_vec = convert_quaternion_to_xyz_orientation(xtraj_attitude[:4,:].T, tsamp)
    traj_positions = xtraj_orbit[:3,:].T

    return np.concatenate([traj_positions, dir_vec, up_vec, right_vec],axis=1), tsamp, np.concatenate([xtraj_orbit.T, xtraj_attitude.T], axis=1)

def convert_pos_to_quaternion(pos_eci):
    # Step 1: Calculate the satellite's direction vector
    zc = direction_vector = - pos_eci / (np.linalg.norm(pos_eci, axis=-1)[..., None])

    # Step 2: Calculate the quaternion orientation
    # Compute the angle between the satellite's local Z-axis and the ECI Z-axis
    north_pole_eci = np.array([0, 0, 1])[None]
    axis_of_rotation_z = np.cross(north_pole_eci, direction_vector)
    rc = axis_of_rotation_z = axis_of_rotation_z / np.linalg.norm(axis_of_rotation_z, axis=-1)[..., None]
    xc = -rc

    # compute the vector pointing to the north from camera
    yc = south_vector = np.cross(rc, zc)
    print(xc.shape, yc.shape, zc.shape)
    R = np.stack([xc, yc, zc], axis=-1)
    rot = transform.Rotation.from_matrix(R)
    quaternion = rot.as_quat()
    return quaternion

def convert_quaternion_to_xyz_orientation(quat, times):
    # Step 1: convert quat to rotation matrix
    # NEED TO SWITCH  QUAT FROM [qw, q1, q2, q3] to [q1, q2, q3, qw]
    quat = np.concatenate([quat[:, 1:], quat[:, :1]], axis=-1)
    rot = transform.Rotation.from_quat(quat)
    R = rot.as_matrix()

    # Step 2: comvert to ECEF from ECI
    Rz = get_Rz(times)
    R = np.matmul(Rz, R)


    # Step 3: compute the x, y, z axis
    xc, yc, zc = R[:, :, 0], R[:, :, 1], R[:, :, 2]
    # xc, yc, zc = R[:, 0], R[:, 1], R[:, 2]
    right_vector = -xc
    up_vector = -yc
    forward_vector = zc

    return forward_vector, up_vector, right_vector 

def get_Rz(times):
    theta_G0_deg = 280.16  # GMST at J2000.0 epoch in degrees
    omega_earth_deg_per_sec = 360 / 86164.100352  # Earth's average rotational velocity in degrees per second
    theta_G_rad = np.deg2rad(theta_G0_deg + omega_earth_deg_per_sec * times)
    ZERO = np.zeros_like(theta_G_rad)
    ONE = np.ones_like(theta_G_rad)

    # Rotation matrix
    Rz = np.stack([
        np.stack([np.cos(theta_G_rad), np.sin(theta_G_rad), ZERO],axis=-1),
        np.stack([-np.sin(theta_G_rad), np.cos(theta_G_rad), ZERO],axis=-1),
        np.stack([ZERO, ZERO, ONE], axis=-1)
    ], axis=-2)
    return Rz

def get_nadir_attitude(x_ecef):
    # Step 1: Calculate the satellite's direction vector
    pos_ecef = x_ecef[:,:3]
    quaternion = np.zeros((pos_ecef.shape[0],4))
    for k in range(pos_ecef.shape[0]):
        zc = direction_vector = - pos_ecef[k,:] / (np.linalg.norm(pos_ecef[k,:]))

        # Step 2: Calculate the quaternion orientation
        # Compute the angle between the satellite's local Z-axis and the ECI Z-axis
        north_pole_eci = np.array([0, 0, 1])
        axis_of_rotation_z = np.cross(north_pole_eci, direction_vector)
        rc = axis_of_rotation_z = axis_of_rotation_z / np.linalg.norm(axis_of_rotation_z)
        xc = -rc

        # compute the vector pointing to the north from camera
        yc = south_vector = np.cross(rc, zc)
        # print(xc.shape, yc.shape, zc.shape)
        R = np.stack([xc, yc, zc], axis=-1)
        rot = transform.Rotation.from_matrix(R)
        quaternion[k,:] = rot.as_quat()
    return quaternion

def get_nadir_attitude_vectors(x_ecef):
    # Step 1: Calculate the satellite's direction vector
    pos_ecef = x_ecef[:,:3]
    quaternion = np.zeros((pos_ecef.shape[0],4))
    zcs = np.zeros((pos_ecef.shape[0],3))
    ycs = np.zeros((pos_ecef.shape[0],3))
    xcs = np.zeros((pos_ecef.shape[0],3))
    for k in range(pos_ecef.shape[0]):
        zc = direction_vector = - pos_ecef[k,:] / (np.linalg.norm(pos_ecef[k,:]))

        # Step 2: Calculate the quaternion orientation
        # Compute the angle between the satellite's local Z-axis and the ECI Z-axis
        north_pole_eci = np.array([0, 0, 1])
        axis_of_rotation_z = np.cross(north_pole_eci, direction_vector)
        rc = axis_of_rotation_z = axis_of_rotation_z / np.linalg.norm(axis_of_rotation_z)
        xc = -rc

        # compute the vector pointing to the north from camera
        yc = south_vector = np.cross(rc, zc)
        # print(xc.shape, yc.shape, zc.shape)
        R = np.stack([xc, yc, zc], axis=-1)
        rot = transform.Rotation.from_matrix(R)
        quaternion[k,:] = rot.as_quat()
        zcs[k,:] = zc
        ycs[k,:] = yc
        xcs[k,:] = xc
    return zcs, -ycs, xcs

def oe_gen(a_min, a_max, i_min=(np.pi/2) - (0.2 * np.random.rand() - 0.1), i_max=(np.pi/2) + (0.2 * np.random.rand() - 0.1)):
    Omega = omega = nu = 2*np.pi*np.random.rand()
    a_min += 6378.0
    a_max += 6378.0
    a = a_min + (a_max - a_min) * np.random.rand()
    e = 0.0 + 0.01 * np.random.rand()
    i = i_min + (i_max - i_min) * np.random.rand()
    return OrbitalElements(a, e, i, Omega, omega, nu)

def get_polar_orbit(a_min=525.0, a_max=575.0):
    oe = oe_gen(a_min, a_max)
    return oe

def get_iss_like_orbit():
    imin = (51.5*np.pi/180)-(0.1*np.random.rand())
    imax = (51.5*np.pi/180)+(0.1*np.random.rand())
    oe = oe_gen(a_min=525.0, a_max=575.0, i_min=imin, i_max=imax)
    return oe

def get_random_orbit(tf=6*60*60):
    if np.random.rand() < 0.5:
        oe = get_iss_like_orbit()
    else:
        oe = get_polar_orbit()
    return generate_eci_traj(oe, tf)
