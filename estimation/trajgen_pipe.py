import numpy as np
import ipdb

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
# ipdb.set_trace()
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

def generate_new_traj(orbit_type='polar'):

    # Use the functions as needed

    # Random initial conditions

    
    if orbit_type == 'polar':
        # Polar Orbit
        oe = oe_polar = OrbitalElements(600.0 + 6378.0 + (100 * np.random.rand() - 50), 0.0 + 0.01 * np.random.rand(), (np.pi / 2) + (0.2 * np.random.rand() - 0.1), 2 * np.pi * np.random.rand(), 2 * np.pi * np.random.rand(), 2 * np.pi * np.random.rand())
        # oe_polar = OrbitalElements(600.0 + 6378.0 + (100 * 0.5 - 50), 0.0 + 0.01 * 0.5, (np.pi / 2) + (0.2 * 0.5 - 0.1), 2 * np.pi * 0.5, 2 * np.pi * 0.5, 2 * np.pi * 0.5)
    else:
        # #ISS~ish Orbit
        oe = oe_iss = OrbitalElements(420.0+6378.0+(100*np.random.rand()-50) ,0.00034+0.01*np.random.rand(), (51.5*np.pi/180)+(0.2*np.random.rand()-0.1), 2*np.pi*np.random.rand(), 2*np.pi*np.random.rand(), 2*np.pi*np.random.rand())

    x0_orbit = oe2eci(oe)

    # Simulate for 3 hours (~2 orbits)
    tf = 3 * 60 * 60
    tsamp = np.arange(0, tf+1, 1)

    xtraj_orbit = np.zeros((6, len(tsamp)))
    xtraj_orbit[:, 0] = x0_orbit

    for k in range(len(tsamp)-1):
        xtraj_orbit[:, k+1] = orbit_step(xtraj_orbit[:, k], 1.0)
    np.set_printoptions(threshold=np. inf, suppress=True, linewidth=np. inf) 
    print(xtraj_orbit[:,:10].T)

    # Random initial conditions
    q0 = np.ones(4)*0.5#np.random.randn(4)
    q0 /= np.linalg.norm(q0)
    omega0 = 2 * (np.pi/180) * np.ones(3)*0.5#np.random.randn(3)
    x0_attitude = np.hstack((q0, omega0))

    # Simulate
    xtraj_attitude = np.zeros((7, len(tsamp)))
    xtraj_attitude[:, 0] = x0_attitude

    for k in range(len(tsamp)-1):
        xtraj_attitude[:, k+1] = attitude_step(xtraj_attitude[:, k], 1.0)

    return np.concatenate([xtraj_orbit, xtraj_attitude], axis=1), tsamp

if __name__ == "__main__":

    # Use the functions as needed

    # Random initial conditions

    # Polar Orbit
    # oe_polar = OrbitalElements(600.0 + 6378.0 + (100 * np.random.rand() - 50), 0.0 + 0.01 * np.random.rand(), (np.pi / 2) + (0.2 * np.random.rand() - 0.1), 2 * np.pi * np.random.rand(), 2 * np.pi * np.random.rand(), 2 * np.pi * np.random.rand())
    oe_polar = OrbitalElements(600.0 + 6378.0 + (100 * 0.5 - 50), 0.0 + 0.01 * 0.5, (np.pi / 2) + (0.2 * 0.5 - 0.1), 2 * np.pi * 0.5, 2 * np.pi * 0.5, 2 * np.pi * 0.5)

    # #ISS~ish Orbit
    # eo_iss = OrbitalElements(420.0+6378.0+(100*np.random.rand()-50) ,0.00034+0.01*np.random.rand(), (51.5*np.pi/180)+(0.2*np.random.rand()-0.1), 2*np.pi*np.random.rand(), 2*np.pi*np.random.rand(), 2*np.pi*np.random.rand())

    x0_orbit = oe2eci(oe_polar)

    # Simulate for 3 hours (~2 orbits)
    tf = 3 * 60 * 60
    tsamp = np.arange(0, tf+1, 1)

    xtraj_orbit = np.zeros((6, len(tsamp)))
    xtraj_orbit[:, 0] = x0_orbit

    for k in range(len(tsamp)-1):
        xtraj_orbit[:, k+1] = orbit_step(xtraj_orbit[:, k], 1.0)
    np.set_printoptions(threshold=np. inf, suppress=True, linewidth=np. inf) 
    print(xtraj_orbit[:,:10].T)

    # Random initial conditions
    q0 = np.ones(4)*0.5#np.random.randn(4)
    q0 /= np.linalg.norm(q0)
    omega0 = 2 * (np.pi/180) * np.ones(3)*0.5#np.random.randn(3)
    x0_attitude = np.hstack((q0, omega0))

    # Simulate
    xtraj_attitude = np.zeros((7, len(tsamp)))
    xtraj_attitude[:, 0] = x0_attitude

    for k in range(len(tsamp)-1):
        xtraj_attitude[:, k+1] = attitude_step(xtraj_attitude[:, k], 1.0)

    print(xtraj_attitude[:,:10].T)