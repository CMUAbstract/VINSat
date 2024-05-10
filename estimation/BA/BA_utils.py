import torch
import numpy as np
from BA.utils import *
from scipy.spatial import transform
from torch_scatter import scatter_sum, scatter_mean

def proj(X, intrinsics):
    """ projection """

    X, Y, Z, W = X.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics.unbind(dim=2) #[...,None,None]

    d = 1.0 / Z.clamp(min=0.1)
    x = fx * (d * X) + cx
    y = fy * (d * Y) + cy

    return torch.stack([x, y], dim=-1)

def attitude_jacobian(q):
    """ attitude jacobian """
    q1, q2, q3, q0 = q.unbind(dim=-1)
    Gq = torch.stack([
        torch.stack([q0, -q3, q2], dim=-1),
        torch.stack([q3, q0, -q1], dim=-1),
        torch.stack([-q2, q1, q0], dim=-1),  
        torch.stack([-q1, -q2, -q3], dim=-1),     
    ], dim=-2)
    return Gq

def landmark_project(poses, landmarks_xyz, intrinsics, ii, jacobian=True):
    poses_ii = poses[:, ii, :]
    bsz = poses_ii.shape[0]
    poses_ii = poses_ii.reshape(-1, 10)[:, :7].requires_grad_(True)
    def project_ii(poses_ii):
        poses_ii = poses_ii.reshape(bsz, -1, 7)
        X1 = apply_inverse_pose_transformation(landmarks_xyz, poses_ii[:, :, 3:], poses_ii[:, :, :3])
        X1 = torch.stack([X1[...,0], X1[...,1], X1[...,2], torch.ones_like(X1[...,1])], dim=-1)
        landmark_est = proj(X1, intrinsics[:, ii])
        landmark_est = landmark_est.reshape(-1, 2)
        return landmark_est
    def project_ii_sum(poses_ii):
        return project_ii(poses_ii).sum(dim=0)
    landmark_est = project_ii(poses_ii).reshape(bsz, -1, 2)
    if jacobian:
        Gq = attitude_jacobian(poses_ii[:,3:])
        Jg = torch.autograd.functional.jacobian(project_ii_sum, poses_ii, vectorize=True).transpose(0,1)
        Jg = torch.cat([Jg[:,:,:3], torch.bmm(Jg[:,:,3:], Gq)], dim=2).reshape(-1, 2, 6)
        Jg = torch.cat([Jg, torch.zeros_like(Jg)[...,:3]], dim=-1)
        return landmark_est, Jg
    return landmark_est

def propagate_orbit_dynamics_skip(position, velocities, times, dt):
    time_diffs = torch.tensor(times[1:] - times[:-1])
    time_diffs = torch.cat([time_diffs, torch.ones_like(time_diffs[-1:])], dim=0)
    max_time_diff = time_diffs.max()
    x = torch.cat([position, velocities], dim=-1)
    x_pred = []
    max_steps_skip = 100
    time_hops = time_diffs//max_steps_skip
    dt = dt*torch.ones_like(x)
    # ipdb.set_trace()
    for i in range(time_hops.max()+1):
        dt = (time_hops == i).float()*(time_diffs%max_steps_skip) + (time_hops > i).float()*max_steps_skip
        dt = dt[None, :, None].expand(-1, -1, 6).to(x)
        x = RK4(x, i, dt)
        x_pred.append(x)
    x_pred = torch.stack(x_pred, dim=-2)
    x_pred = x_pred[..., torch.arange(len(time_diffs)), time_hops, :]
    pos_pred = x_pred[..., :3]
    vel_pred = x_pred[..., 3:]
    return pos_pred, vel_pred

def propagate_orbit_dynamics(position, velocities, times, dt):
    time_diffs = torch.tensor(times[1:] - times[:-1])
    time_diffs = torch.cat([time_diffs, torch.ones_like(time_diffs[-1:])], dim=0)
    max_time_diff = time_diffs.max()
    x = torch.cat([position, velocities], dim=-1)
    x_pred = []
    dt = dt*torch.ones_like(x)
    for i in range(max_time_diff):
        x = RK4(x, i, dt)
        x_pred.append(x)
    x_pred = torch.stack(x_pred, dim=-2)
    x_pred = x_pred[..., torch.arange(len(time_diffs)), time_diffs-1, :]
    pos_pred = x_pred[..., :3]
    vel_pred = x_pred[..., 3:]
    return pos_pred, vel_pred

def propagate_orbit_dynamics_init(position, velocities, duration, dt, only_end=False):
    x = torch.cat([position, velocities], dim=-1)
    x_pred = [x]
    dt = dt*torch.ones_like(x)
    for i in range(duration):
        x = RK4(x, i, dt)
        x_pred.append(x)
    x_pred = torch.stack(x_pred, dim=-2)
    if only_end:
        x_pred = x_pred[..., -1, :]
    pos_pred = x_pred[..., :3]
    vel_pred = x_pred[..., 3:]
    return pos_pred, vel_pred

def propagate_rotation_dynamics_init(quaternion, omegas, duration, dt, only_end=False):
    q_pred = [quaternion]
    for i in range(duration):
        quaternion_out = quaternion_multiply(quaternion, quaternion_exp(dt * omegas[:, i]))
        q_pred.append(quaternion_out)
        quaternion = quaternion_out
    q_pred = torch.stack(q_pred, dim=-2)
    if only_end:
        q_pred = q_pred[..., -1, :]
    return q_pred

def propagate_dynamics_init(states, velocities, omega, tdiff, duration, dt):
    position, rotation = states[:,:3], states[:,3:7]
    # concate the 4 corners of the hessian into a matrix 
    w = omega#[..., :3]#, imu_meas[..., 3:]

    position_beg, velocities_beg = propagate_orbit_dynamics_init(position, velocities, tdiff, dt, only_end=False)
    quat_beg = propagate_rotation_dynamics_init(rotation, w[:, :tdiff, :], tdiff, dt, only_end=False)

    position_t, velocities_t = propagate_orbit_dynamics_init(position_beg[:, -1], velocities_beg[:, -1], duration, dt, only_end=False)
    quat_t = propagate_rotation_dynamics_init(quat_beg[:, -1], w[:, tdiff:, :], duration, dt, only_end=False)

    states_beg = torch.cat([position_beg, quat_beg, velocities_beg], dim=-1)[:, 1:-1]
    states_t = torch.cat([position_t, quat_t, velocities_t], dim=-1)
    states_full = torch.cat([states_beg, states_t], dim=-2)
    velocities_full = torch.cat([velocities_beg, velocities_t], dim=-2)[:, 1:-1]
    return states_t, velocities_t, states_full, velocities_full



def compute_orbit_jacobian(x, i, dt):
    def orbit_prop(x):
        return RK4(x, i, dt).sum(dim=0).reshape(-1)
    return torch.autograd.functional.jacobian(orbit_prop, (x), vectorize=True).reshape(-1, 6, 6)

def propagate_orbit_dynamics_cov_init(position, velocities, duration, dt, sigma, Q=0, only_end=False):
    x = torch.cat([position, velocities], dim=-1)
    x_pred = [x]
    sigma_pred = [sigma]
    dt = dt*torch.ones_like(x)
    for i in range(duration):
        # Compute the Jacobian J_pos_vel of the dynamics
        J_pos_vel = compute_orbit_jacobian(x, i, dt)  
        x = RK4(x, i, dt)
        sigma = J_pos_vel @ sigma @ J_pos_vel.transpose(-1,-2) + Q
        x_pred.append(x)
        sigma_pred.append(sigma)
    x_pred = torch.stack(x_pred, dim=-2)
    sigma_pred = torch.stack(sigma_pred, dim=-3)
    if only_end:
        x_pred = x_pred[..., -1, :]
        sigma_pred = sigma_pred[..., -1, :, :]
    pos_pred = x_pred[..., :3]
    vel_pred = x_pred[..., 3:]
    return pos_pred, vel_pred, sigma_pred

def hat(v):
    """
    Compute the hat (skew-symmetric) operator for a batch of vectors.
    :param v: Tensor with shape [batch_size, 3]
    :return: Tensor with shape [batch_size, 3, 3]
    """
    zero = torch.zeros(v.shape, dtype=v.dtype, device=v.device)[...,:1]
    vhat = torch.cat((zero, -v[..., 2:3], v[..., 1:2],
                        v[..., 2:3], zero, -v[..., 0:1],
                        -v[..., 1:2], v[..., 0:1], zero), dim=-1)#.view(-1, 3, 3)
    vhat_shape = list(v.shape[:-1]) + [3, 3]
    return vhat.view(vhat_shape)

def L(q):
    """
    Compute the L matrix for a batch of quaternions.
    :param q: Tensor with shape [batch_size, 4]
    :return: Tensor with shape [batch_size, 4, 4]
    """
    s = q[..., 0:1]
    v = q[..., 1:]
    v_hat = hat(v)
    sI_plus_v_hat = torch.eye(3, dtype=q.dtype, device=q.device) * s.unsqueeze(-1) + v_hat
    L_mat = torch.cat((torch.cat((s, -v), dim=-1).unsqueeze(-2), torch.cat((v.unsqueeze(-1), sI_plus_v_hat), dim=-1)), dim=-2)
    return L_mat#.view(-1, 4, 4)

def qtoQ(q):
    """
    Convert a batch of quaternions to the Q matrices.
    :param q: Tensor with shape [batch_size, 4]
    :return: Tensor with shape [batch_size, 3, 3]
    """
    Tshape = list(q.shape[:-1]) + [4, 4]
    Hshape = list(q.shape[:-1]) + [4, 3]
    T = torch.diag_embed(torch.tensor([1, -1, -1, -1], dtype=q.dtype, device=q.device)).expand(Tshape) #q.shape[0], 4, 4)
    H = torch.cat((torch.zeros((1, 3), dtype=q.dtype, device=q.device), torch.eye(3, dtype=q.dtype, device=q.device)), dim=-2).expand(Hshape)
    
    L_q = L(q)
    # return H.t()*T*L(q)*T*L(q)*H
    intermed = torch.matmul(torch.matmul(T, L_q), torch.matmul(T, L_q))
    Q = torch.matmul(torch.matmul(H.transpose(-2, -1), intermed), H)
    return Q#[..., 1:, 1:]  # Extracting the bottom right 3x3 block

def compute_rot_jacobian(omega, dt):
    dq = quaternion_exp(-dt * omega)
    Y = qtoQ(dq)
    return Y
    
def propagate_rotation_dynamics_cov_init(quaternion, omegas, duration, dt, sigma_rot, Q_rot=0, only_end=False):
    q_pred = [quaternion]
    sigma_q_pred = [sigma_rot]
    for i in range(duration):
        # Compute the Jacobian J_rot of the rotation dynamics
        J_rot = compute_rot_jacobian(omegas[:, i], dt)  
        quaternion_out = quaternion_multiply(quaternion, quaternion_exp(dt * omegas[:, i]))
        # ipdb.set_trace()
        sigma_rot = J_rot @ sigma_rot @ J_rot.transpose(-1,-2) + Q_rot
        q_pred.append(quaternion_out)
        sigma_q_pred.append(sigma_rot)
        quaternion = quaternion_out
    q_pred = torch.stack(q_pred, dim=-2)
    sigma_q_pred = torch.stack(sigma_q_pred, dim=-3)
    if only_end:
        q_pred = q_pred[..., -1, :]
        sigma_q_pred = sigma_q_pred[..., -1, :, :]
    return q_pred, sigma_q_pred


def propagate_dynamics_cov_init(states, velocities, hessian, omega, tdiff, duration, dt):
    position, rotation = states[:,:3], states[:,3:7]
    # concate the 4 corners of the hessian into a matrix 
    hessian = hessian.reshape(-1, 9, 9)
    hessian_pos_pos, hessian_pos_vel, hessian_vel_vel, hessian_vel_pos = hessian[:, :3, :3], hessian[:, :3, 6:], hessian[:, 6:, 6:], hessian[:, 6:, :3]
    hessian_state = torch.cat([torch.cat([hessian_pos_pos, hessian_pos_vel], dim=-1), torch.cat([hessian_vel_pos, hessian_vel_vel], dim=-1)], dim=-2)
    hessian_rot = hessian[:, 3:6, 3:6]
    cov_rot = torch.inverse(hessian_rot)
    cov_state = torch.inverse(hessian_state)
    w = omega#[..., :3]#, imu_meas[..., 3:]

    position_beg, velocities_beg, cov_state_beg = propagate_orbit_dynamics_cov_init(position, velocities, tdiff, dt, cov_state, only_end=True)
    quat_beg, cov_rot_beg = propagate_rotation_dynamics_cov_init(rotation, w[:, :tdiff, :], tdiff, dt, cov_rot, only_end=True)

    ipdb.set_trace()
    position_t, velocities_t, cov_state_t = propagate_orbit_dynamics_cov_init(position_beg, velocities_beg, duration, dt, cov_state_beg, only_end=False)
    quat_t, cov_rot_t = propagate_rotation_dynamics_cov_init(quat_beg, w[:, tdiff:, :], duration, dt, cov_rot_beg, only_end=False)

    states_t = torch.cat([position_t, quat_t, velocities_t], dim=-1)
    hessian_state_t = torch.inverse(cov_state_t)
    hessian_rot_t = torch.inverse(cov_rot_t)
    return states_t, velocities_t, hessian_state_t, hessian_rot_t


def propagate_rotation_dynamics(quaternion, omegas, times, dt):#, jac=False):
    # ipdb.set_trace()
    time_diffs = torch.tensor(times[1:] - times[:-1])
    time_diffs = torch.cat([time_diffs, torch.ones_like(time_diffs[-1:])], dim=0)
    max_time_diff = time_diffs.max()
    q_pred = []
    bsz, N = quaternion.shape[:2]
    jac_qpred = torch.eye(4)[None, None].repeat(bsz, N, 1, 1).double()
    # quaternion = quaternion.reshape(-1, 4)
    # omegas = omegas.reshape(-1, len(max_time_diff), 3)
    # ipdb.set_trace()
    for i in range(max_time_diff):
        quaternion_out = quaternion_multiply(quaternion, quaternion_exp(dt * omegas[:, :, i]))
        # if jac:
        #     mask = (i < time_diffs)#[None, :, None, None]
        #     jac_i = quaternion_jacobian(quaternion_exp(dt * omegas[:, :, i]))
        #     jac_qpred[:, mask] = (jac_i[:,mask][..., None]*jac_qpred[:, mask, None]).sum(dim=-2)
        q_pred.append(quaternion_out)
        quaternion = quaternion_out
    
    q_pred = torch.stack(q_pred, dim=-2)
    # q_pred = q_pred.reshape(bsz, -1, len(max_time_diff), 4)
    q_pred = q_pred[:, torch.arange(len(time_diffs)), time_diffs-1]#torch.zeros_like(time_diffs-1)]
    # pos_pred = x_pred[:, :, :3]
    # vel_pred = x_pred[:, :, 3:]
    return q_pred, jac_qpred

def precompute_cum_rotations(omegas, dt):
    # Step 1: Compute the exponential of all the omegas
    rotations = quaternion_exp(dt * omegas)
    
    # Step 2: Compute the cumulative product of these rotations
    cum_rotations = [rotations[:, :, 0]]
    for i in range(1, rotations.shape[2]):
        cum_rot = quaternion_multiply(cum_rotations[-1], rotations[:, :, i])
        cum_rotations.append(cum_rot)
    cum_rotations = torch.stack(cum_rotations, dim=-2)
    return cum_rotations

def propagate_rotation_dynamics_precomp(quaternion, cum_rotations, times, dt):
    time_diffs = torch.tensor(times[1:] - times[:-1])
    time_diffs = torch.cat([time_diffs, torch.ones_like(time_diffs[-1:])], dim=0)

    # cum_rotation = precompute_cum_rotations(omegas, dt)[:,:,-1]
    cum_rotation = cum_rotations[:, :, -1]
    # Step 3: Multiply the initial quaternion by the aggregate rotations
    q_pred = quaternion_multiply(quaternion, cum_rotation)
    # q_pred = q_pred[:, torch.arange(len(time_diffs)), time_diffs-1]
    
    # The Jacobian remains the same as in your original code
    bsz, N = quaternion.shape[:2]
    jac_qpred = torch.eye(4)[None, None].repeat(bsz, N, 1, 1).double()
    
    return q_pred, jac_qpred

def predict_GN(poses, velocities, imu_meas, times, quat_coeff, dt=1, jacobian=True):
    w, a = imu_meas[..., :3], imu_meas[..., 3:]
    # phi = quaternion_log(poses[:,:,3:])
    # position = poses[:,:,:3]
    # rotation = poses[:,:,3:]
    # vel = velocities + dt * (apply_pose_transformation_quat(a.unsqueeze(-1), rotation)).squeeze(-1) # SO3(rotation).act(a.unsqueeze(-1)).squeeze(-1)
    # pos_pred = position + dt * vel
    # phi_pred = phi + dt * w
    # q_pred = quaternion_exp(phi_pred)#.data  # TODO: why .data?
    # pose_pred = torch.cat([pos_pred, quaternion_exp(phi_pred)], 2) # TODO: why .data? for quaternion_exp().data
    # vel_pred = vel
    # res_pred = torch.cat([pos_pred[:,:-1] - position[:,1:], 1 - torch.abs(q_pred*rotation[:,1:])], 2)
    # velocities = velocities[:, times[:]:]
    # w = w[:, times[:]]
    bsz = poses.shape[0]
    N = poses.shape[1]
    num_res = (N-1)*4
    def res_preds(poses, jac=False):
        # phi = quaternion_log(poses[:,:,3:])
        # ipdb.set_trace()
        position = poses[:,:,:3]
        rotation = poses[:,:,3:]
        # pos_pred, vel_pred = propagate_dynamics(position, velocities, times, dt)
        vel = velocities + dt * a#apply_pose_transformation_quat(a, rotation) # SO3(rotation).act(a.unsqueeze(-1)).squeeze(-1)
        pos_pred = position + dt * velocities.sum(dim=-2)
        q_pred, jac_qpred = propagate_rotation_dynamics(rotation, w, times, dt, jac)
        jac_ppred = torch.eye(3,3)[None, None].repeat(bsz, N, 1, 1)
        # ipdb.set_trace()
        # q_pred = quaternion_multiply(poses[:,:,3:], quaternion_exp(dt * w))#.data  # TODO: why .data?
        # phi_pred = phi + dt * w.sum(dim=-2)
        # q_pred = quaternion_exp(phi_pred)#.data  # TODO: why .data?
        pose_pred = torch.cat([pos_pred, q_pred], 2) # TODO: why .data? for quaternion_exp().data
        vel_pred = vel
        res_pred = torch.cat([(pos_pred[:,:-1] - position[:,1:])*0, quat_coeff*(1 - 1*torch.abs(q_pred[:,:-1]*rotation[:,1:]).sum(dim=-1).unsqueeze(-1))], 2)
        qthat = q_pred[:,:-1]
        qt = rotation[:,1:]
        qt1 = rotation[:,:-1]
        # ipdb.set_trace()
        return res_pred, pose_pred, vel_pred, quat_coeff*torch.abs(q_pred[:,:-1]*rotation[:,1:]).sum(dim=-1), jac_ppred, jac_qpred, qt, qt1, qthat
    def res_preds_sum(poses):
        poses = poses.reshape(bsz, -1, 7)
        return res_preds(poses)[0].sum(dim=0).reshape(-1)
    def res_preds_sum_grad(poses):
        poses = poses.reshape(bsz, -1, 7)
        Gq = attitude_jacobian(poses[:,:,3:])
        res_out = res_preds(poses)[0][:,:,-1].sum()
        res_grad = torch.autograd.grad(res_out, poses, create_graph=True)[0]
        res_grad = torch.cat([res_grad[:, :, :3], (res_grad[:, :, 3:, None]*Gq).sum(dim=2)], dim=2)
        return res_grad.reshape(-1)
    res_pred, pose_pred, vel_pred, qdot, jac_ppred, jac_qpred, qt, qt1, qthat = res_preds(poses, jacobian)
    if jacobian:
        # Jf = torch.zeros(4*poses.shape[0], 3*poses.shape[1])
        # Jf[0::4, 0::3] = torch.eye(poses.shape[0])
        Gq = attitude_jacobian(poses[:,:,3:])
        Jf = torch.autograd.functional.jacobian(res_preds_sum, poses.reshape(bsz, -1), vectorize=True).reshape(bsz, -1, 7)
        # Jf = torch.zeros(bsz, num_res*N, 7)
        GqJ = Gq[:, None].repeat(bsz, num_res, 1, 1, 1).reshape(bsz, -1, 4, 3)
        Jf = torch.cat([Jf[:,:,:3], (Jf[:,:,3:,None] * GqJ/2).sum(dim=2)], dim=2)
        Jf = Jf.reshape(bsz, num_res, N*6)

        Hdiff = torch.autograd.functional.jacobian(res_preds_sum_grad, poses[:, :, :].reshape(bsz, -1), vectorize=True).reshape(bsz, N, 6, N, 7)
        # ipdb.set_trace()
        Hq_full = Hdiff = torch.cat([Hdiff[..., :3], (Hdiff[..., 3:, None]*Gq[:,None,None]).sum(dim=-2)], dim=-1).reshape(bsz, N, 6, N, 6)
        # Hdiff = (Hdiff[..., 3:, None]*Gq[:,None,None]).sum(dim=-2).reshape(bsz, N, 3, N, 3)
        # qdotzero = torch.zeros_like(qdot[:, :1])
        # qdotsign = torch.sign(qdot)
        # Hqtt = torch.cat([qdotzero, qdotsign*qdot], dim=1)[:, :, None, None]*torch.eye(3, 3)[None, None]
        # Hqt1t1 = (qt*(jac_qpred[:,:-1]*qt1[:, :, None, :]).sum(dim=-1)).sum(dim=-1)
        # Hqt1t1 = torch.eye(3, 3)[None, None]*torch.cat([qdotsign*Hqt1t1, qdotzero], dim=1)[:,:,None,None]
        # Hqtt1 = -qdotsign[...,None, None]*(Gq[:, :-1].transpose(-1, -2)[...,None]*(jac_qpred[:, :-1].transpose(-1, -2)[...,None]*Gq[:, 1:, None]).sum(dim=-2)[:, :, None]).sum(dim=-2)
        # # Hqt1t = Hqtt1.transpose(-1, -2)
        # # Hq2 = torch.eye(3, 3)[None, None]*torch.cat([qdotabs, qdotzero], dim=1)[:,:,None,None]
        # # Hq = Hq1 + Hq2
        # Hqdiag = torch.block_diag(*(Hqtt + Hqt1t1)[0].unbind(dim=0)).unsqueeze(0)
        # Hqtt1diag = torch.block_diag(*(Hqtt1)[0].unbind(dim=0)).unsqueeze(0)
        # # Hqt1tdiag = torch.block_diag(*(Hqt1t)[0].unbind(dim=0)).unsqueeze(0)
        # Hqtt1diag = torch.cat([Hqtt1diag, torch.zeros_like(Hqtt1diag)[:,:,:3]], dim=2)
        # Hqtt1diag = torch.cat([torch.zeros_like(Hqtt1diag)[:,:3], Hqtt1diag], dim=1)
        # Hqt1tdiag = Hqtt1diag.transpose(-1, -2)
        # Hq = (Hqdiag + Hqtt1diag + Hqt1tdiag).reshape(bsz, N, 3, N, 3)
        # Hq_full = torch.zeros(bsz, N, 6, N, 6)
        # Hq_full[:, :, :3, :, :3] = Hq
        Hq_full = Hq_full.reshape(bsz, N*6, N*6)
        # ipdb.set_trace()

        # Jf = Jf.reshape(bsz, (N-1), num_res/(N-1), N, 6)[:, :, :3, :, :3].reshape(bsz, (N-1)*3, N*3)
        return res_pred, pose_pred, vel_pred, 0, 0, Jf, Hq_full
    
    return res_pred, pose_pred, vel_pred

def predict_vel(poses, velocities, imu_meas, times, quat_coeff, dt=1, jacobian=True):
    w, a = imu_meas[..., :3], imu_meas[..., 3:]
    bsz = poses.shape[0]
    N = poses.shape[1]
    num_res = (N-1)*3
    def res_preds(poses, jac=False):
        position = poses[:,:,:3]
        rotation = poses[:,:,3:]
        # pos_pred, vel_pred = propagate_dynamics(position, velocities, times, dt)
        vel = velocities + dt * a#apply_pose_transformation_quat(a, rotation) # SO3(rotation).act(a.unsqueeze(-1)).squeeze(-1)
        pos_pred = position + dt * velocities.sum(dim=-2)
        q_pred, jac_qpred = propagate_rotation_dynamics(rotation, w, times, dt, jac)
        jac_ppred = torch.eye(3,3)[None, None].repeat(bsz, N, 1, 1)
        pose_pred = torch.cat([pos_pred, q_pred], 2) 
        vel_pred = vel
        res_pred = torch.cat([(pos_pred[:,:-1] - position[:,1:]), quat_coeff*(1 - torch.abs((q_pred[:,:-1]*rotation[:,1:]).sum(dim=-1)).unsqueeze(-1))], 2)
        qthat = q_pred[:,:-1]
        qt = rotation[:,1:]
        qt1 = rotation[:,:-1]
        # print(res_pred[:, :, -1].abs().mean(), (q_pred[0,:-1]*poses[0,1:,3:]).sum(dim=-1).mean(), (q_pred[0,:-1]*rotation[0,1:]).sum(dim=-1).mean())
        return res_pred, pose_pred, vel_pred, quat_coeff*torch.abs((q_pred[:,:-1]*rotation[:,1:]).sum(dim=-1)), jac_ppred, jac_qpred, qt, qt1, qthat
    def res_preds_sum(poses):
        poses = poses.reshape(bsz, -1, 7)
        return res_preds(poses)[0].sum(dim=0)[:,:-1].reshape(-1)
    def res_preds_sum_quat(poses):
        poses = poses.reshape(bsz, -1, 7)
        return res_preds(poses)[0].sum(dim=0)[:,-1].reshape(-1)
    def res_preds_sum_grad(poses):
        poses = poses.reshape(bsz, -1, 7)
        Gq = attitude_jacobian(poses[:,:,3:])
        res_out = res_preds(poses)[0][:,:,-1].sum()
        res_grad = torch.autograd.grad(res_out, poses, create_graph=True)[0]
        res_grad = torch.cat([res_grad[:, :, :3], (res_grad[:, :, 3:, None]*Gq).sum(dim=2)], dim=2)
        return res_grad.reshape(-1)
    res_pred, pose_pred, vel_pred, qdot, jac_ppred, jac_qpred, qt, qt1, qthat = res_preds(poses, jacobian)
    if jacobian:
        # print(res_pred[:, :, -1].abs().mean(), (pose_pred[0,:-1,3:]*poses[0,1:,3:]).sum(dim=-1).mean())
        Gq = attitude_jacobian(poses[:,:,3:])
        Jf = torch.autograd.functional.jacobian(res_preds_sum, poses.reshape(bsz, -1), vectorize=True).reshape(bsz, -1, 7)
        GqJ = Gq[:, None].repeat(bsz, num_res, 1, 1, 1).reshape(bsz, -1, 4, 3)
        Jf = torch.cat([Jf[:,:,:3], (Jf[:,:,3:,None] * GqJ).sum(dim=2)], dim=2)
        Jf = Jf.reshape(bsz, num_res, N*6)

        if False:
            Jf_quat = torch.autograd.functional.jacobian(res_preds_sum_quat, poses.reshape(bsz, -1), vectorize=True).reshape(bsz, -1, 7)
            GqJ = Gq[:, None].repeat(bsz, N-1, 1, 1, 1).reshape(bsz, -1, 4, 3)
            Jf_quat = torch.cat([Jf_quat[:,:,:3], (Jf_quat[:,:,3:,None] * GqJ).sum(dim=2)], dim=2)
            Jf_quat = Jf_quat.reshape(bsz, N-1, N*6)
            qgrad = (Jf_quat*res_pred[:, :, -1:]).sum(dim=1).reshape(bsz, N, 6)
            Hq_full = torch.bmm(Jf_quat.transpose(-1, -2), Jf_quat)
        else:
            poses.requires_grad_(True)
            qgrad = res_preds_sum_grad(poses).reshape(bsz, N, 6)
            Hdiff = torch.autograd.functional.jacobian(res_preds_sum_grad, poses[:, :, :].reshape(bsz, -1), vectorize=True).reshape(bsz, N, 6, N, 7)
            # ipdb.set_trace()
            Hq_full = torch.cat([Hdiff[..., :3], (Hdiff[..., 3:, None]*Gq[:,None,None]).sum(dim=-2)], dim=-1).reshape(bsz, N, 6, N, 6)
            Hq_full = Hq_full.reshape(bsz, N*6, N*6)
        return res_pred, pose_pred, vel_pred, 0, 0, Jf, Hq_full, qgrad
    
    return res_pred, pose_pred, vel_pred

def predict(states, imu_meas, times, quat_coeff, vel_coeff, dt=1, jacobian=True, initialize=False):
    w, a, cum_rots = imu_meas[..., :3], imu_meas[..., 3:6], imu_meas[..., 6:]
    bsz = states.shape[0]
    N = states.shape[1]
    num_res = (N-1)*6
    GN_quat = False
    if initialize:
        if jacobian:
            return torch.zeros((bsz, N-1, 6)), 0, 0, 0, 0, torch.zeros((bsz, num_res, N*9)), torch.zeros((bsz, N*9, N*9)), torch.zeros((bsz, N, 9))
        return torch.zeros((bsz, N-1, 6)), 0, 0
    def res_preds(states):
        position = states[:,:,:3]
        rotation = states[:,:,3:7]
        velocities = states[:,:,7:]
        pos_pred, vel_pred = propagate_orbit_dynamics(position, velocities, times, dt)
        # q_pred, jac_qpred = propagate_rotation_dynamics(rotation, w, times, dt)#, jac)
        q_pred, _ = propagate_rotation_dynamics_precomp(rotation, cum_rots, times, dt)
        jac_ppred = torch.eye(3,3)[None, None].repeat(bsz, N, 1, 1)
        state_pred = torch.cat([pos_pred, q_pred, vel_pred], 2) 
        res_pred = torch.cat([(pos_pred[:,:-1] - position[:,1:]), (vel_pred[:,:-1]-velocities[:,1:])*vel_coeff, quat_coeff*(1 - torch.abs((q_pred[:,:-1]*rotation[:,1:]).sum(dim=-1)).unsqueeze(-1))], 2)
        qthat = q_pred[:,:-1]
        qt = rotation[:,1:]
        qt1 = rotation[:,:-1]
        return res_pred, state_pred, vel_pred
    def res_preds_quat_only(states):
        position = states[:,:,:3]
        rotation = states[:,:,3:7]
        velocities = states[:,:,7:]
        # q_pred, jac_qpred = propagate_rotation_dynamics(rotation, w, times, dt)#, jac)
        q_pred, _ = propagate_rotation_dynamics_precomp(rotation, cum_rots, times, dt)
        return quat_coeff*(1 - torch.abs((q_pred[:,:-1]*rotation[:,1:]).sum(dim=-1)).unsqueeze(-1))
    def res_preds_sum(states):
        states = states.reshape(bsz, -1, 10)
        return res_preds(states)[0].sum(dim=0)[:,:-1].reshape(-1)
    def res_preds_sum_quat(states):
        states = states.reshape(bsz, -1, 10)
        return res_preds(states)[0].sum(dim=0)[:,-1].reshape(-1)
    def res_preds_sum_grad(states):
        states = states.reshape(bsz, -1, 10)
        Gq = attitude_jacobian(states[:,:,3:7])
        res_out = res_preds_quat_only(states).sum()#, quat_out_only=True
        res_grad = torch.autograd.grad(res_out, states, create_graph=True)[0]
        res_grad = torch.cat([res_grad[:, :, :3], (res_grad[:, :, 3:7, None]*Gq).sum(dim=2), res_grad[:, :, 7:]], dim=2)
        return res_grad.reshape(-1)
    with torch.no_grad():
        res_pred, pose_pred, vel_pred = res_preds(states)#, jacobian)
    # print("finished dynamics propagation")
    if jacobian:
        Gq = attitude_jacobian(states[:,:,3:7])
        Jf = torch.autograd.functional.jacobian(res_preds_sum, states.reshape(bsz, -1), vectorize=True).reshape(bsz, -1, 10)
        GqJ = Gq[:, None].repeat(bsz, num_res, 1, 1, 1).reshape(bsz, -1, 4, 3)
        Jf = torch.cat([Jf[:,:,:3], (Jf[:,:,3:7,None] * GqJ).sum(dim=2), Jf[:,:,7:]], dim=2)
        Jf = Jf.reshape(bsz, num_res, N*9)
        # print("finished jacobian computation")
        if GN_quat:
            Jf_quat = torch.autograd.functional.jacobian(res_preds_sum_quat, states.reshape(bsz, -1), vectorize=True).reshape(bsz, -1, 7)
            GqJ = Gq[:, None].repeat(bsz, N-1, 1, 1, 1).reshape(bsz, -1, 4, 3)
            Jf_quat = torch.cat([Jf_quat[:,:,:3], (Jf_quat[:,:,3:,None] * GqJ).sum(dim=2)], dim=2)
            Jf_quat = Jf_quat.reshape(bsz, N-1, N*6)
            qgrad = (Jf_quat*res_pred[:, :, -1:]).sum(dim=1).reshape(bsz, N, 6)
            Hq_full = torch.bmm(Jf_quat.transpose(-1, -2), Jf_quat)
        else:
            states.requires_grad_(True)
            qgrad = res_preds_sum_grad(states).reshape(bsz, N, 9)
            Hdiff = torch.autograd.functional.jacobian(res_preds_sum_grad, states[:, :, :].reshape(bsz, -1), vectorize=True).reshape(bsz, N, 9, N, 10)
            Hq_full = torch.cat([Hdiff[..., :3], (Hdiff[..., 3:7, None]*Gq[:,None,None]).sum(dim=-2), Hdiff[..., 7:]], dim=-1).reshape(bsz, N, 9, N, 9)
            Hq_full = Hq_full.reshape(bsz, N*9, N*9)
        # print("finished hessian computation")
        return res_pred, pose_pred, vel_pred, 0, 0, Jf, Hq_full, qgrad
    
    return res_pred, pose_pred, vel_pred

def predict_gpu(states, imu_meas, times, quat_coeff, vel_coeff, dt=1, jacobian=True, initialize=False):
    w, a, cum_rots = imu_meas[..., :3].cuda(), imu_meas[..., 3:6], imu_meas[..., 6:].cuda()
    bsz = states.shape[0]
    N = states.shape[1]
    num_res = (N-1)*6
    GN_quat = False
    if initialize:
        if jacobian:
            return torch.zeros((bsz, N-1, 6)), 0, 0, 0, 0, torch.zeros((bsz, num_res, N*9)), torch.zeros((bsz, N*9, N*9)), torch.zeros((bsz, N, 9))
        return torch.zeros((bsz, N-1, 6)), 0, 0
    def res_preds(states):
        position = states[:,:,:3]
        rotation = states[:,:,3:7]
        velocities = states[:,:,7:]
        # pos_pred, vel_pred = propagate_orbit_dynamics(position, velocities, times, dt)
        pos_pred, vel_pred = propagate_orbit_dynamics_skip(position, velocities, times, dt)
        # q_pred, jac_qpred = propagate_rotation_dynamics(rotation, w, times, dt)#, jac)
        q_pred, _ = propagate_rotation_dynamics_precomp(rotation, cum_rots, times, dt)#, jac)
        # ipdb.set_trace()
        jac_ppred = torch.eye(3,3)[None, None].repeat(bsz, N, 1, 1)
        state_pred = torch.cat([pos_pred, q_pred, vel_pred], 2) 
        res_pred = torch.cat([(pos_pred[:,:-1] - position[:,1:]), (vel_pred[:,:-1]-velocities[:,1:])*vel_coeff, quat_coeff*(1 - torch.abs((q_pred[:,:-1]*rotation[:,1:]).sum(dim=-1)).unsqueeze(-1))], 2)
        qthat = q_pred[:,:-1]
        qt = rotation[:,1:]
        qt1 = rotation[:,:-1]
        return res_pred, state_pred, vel_pred
    def res_preds_quat_only(states):
        position = states[:,:,:3]
        rotation = states[:,:,3:7]
        velocities = states[:,:,7:]
        # q_pred, jac_qpred = propagate_rotation_dynamics(rotation, w, times, dt)#, jac)
        q_pred, _ = propagate_rotation_dynamics_precomp(rotation, cum_rots, times, dt)#, jac)
        return quat_coeff*(1 - torch.abs((q_pred[:,:-1]*rotation[:,1:]).sum(dim=-1)).unsqueeze(-1))
    def res_preds_sum(states):
        states = states.reshape(bsz, -1, 10)
        return res_preds(states)[0].sum(dim=0)[:,:-1].reshape(-1)
    def res_preds_sum_quat(states):
        states = states.reshape(bsz, -1, 10)
        return res_preds(states)[0].sum(dim=0)[:,-1].reshape(-1)
    def res_preds_sum_grad(states):
        states = states.reshape(bsz, -1, 10)
        Gq = attitude_jacobian(states[:,:,3:7])
        res_out = res_preds_quat_only(states).sum()#, quat_out_only=True
        res_grad = torch.autograd.grad(res_out, states, create_graph=True)[0]
        res_grad = torch.cat([res_grad[:, :, :3], (res_grad[:, :, 3:7, None]*Gq).sum(dim=2), res_grad[:, :, 7:]], dim=2)
        return res_grad.reshape(-1)
    with torch.no_grad():
        res_pred, pose_pred, vel_pred = res_preds(states.cuda())#, jacobian)
        res_pred = res_pred.cpu()
    # print("finished dynamics propagation")
    if jacobian:
        Gq = attitude_jacobian(states[:,:,3:7])
        Jf = torch.autograd.functional.jacobian(res_preds_sum, states.reshape(bsz, -1).cuda(), vectorize=True).reshape(bsz, -1, 10).cpu()
        GqJ = Gq[:, None].repeat(bsz, num_res, 1, 1, 1).reshape(bsz, -1, 4, 3)
        Jf = torch.cat([Jf[:,:,:3], (Jf[:,:,3:7,None] * GqJ).sum(dim=2), Jf[:,:,7:]], dim=2)
        Jf = Jf.reshape(bsz, num_res, N*9)
        # print("finished jacobian computation")
        if GN_quat:
            Jf_quat = torch.autograd.functional.jacobian(res_preds_sum_quat, states.reshape(bsz, -1), vectorize=True).reshape(bsz, -1, 7)
            GqJ = Gq[:, None].repeat(bsz, N-1, 1, 1, 1).reshape(bsz, -1, 4, 3)
            Jf_quat = torch.cat([Jf_quat[:,:,:3], (Jf_quat[:,:,3:,None] * GqJ).sum(dim=2)], dim=2)
            Jf_quat = Jf_quat.reshape(bsz, N-1, N*6)
            qgrad = (Jf_quat*res_pred[:, :, -1:]).sum(dim=1).reshape(bsz, N, 6)
            Hq_full = torch.bmm(Jf_quat.transpose(-1, -2), Jf_quat)
        else:
            states.requires_grad_(True)
            qgrad = res_preds_sum_grad(states.cuda()).reshape(bsz, N, 9).cpu()
            Hdiff = torch.autograd.functional.jacobian(res_preds_sum_grad, states[:, :, :].reshape(bsz, -1).cuda(), vectorize=True).reshape(bsz, N, 9, N, 10).cpu()
            Hq_full = torch.cat([Hdiff[..., :3], (Hdiff[..., 3:7, None]*Gq[:,None,None]).sum(dim=-2), Hdiff[..., 7:]], dim=-1).reshape(bsz, N, 9, N, 9)
            Hq_full = Hq_full.reshape(bsz, N*9, N*9)
        # print("finished hessian computation")
        return res_pred, pose_pred, vel_pred, 0, 0, Jf, Hq_full, qgrad
    
    return res_pred, pose_pred, vel_pred

def prior_gpu(states, prop_states, vel_coeff, quat_coeff, hessian_state_t, hessian_rot_t, jacobian=True, initialize=False):
    bsz = states.shape[0]
    N = states.shape[1]
    prop_states = prop_states.cuda()
    num_res = (N)*6
    GN_quat = False
    if initialize:
        if jacobian:
            return torch.zeros((bsz, N, 6)), torch.zeros((bsz, num_res, N*9)), torch.zeros((bsz, N*9, N*9)), torch.zeros((bsz, N, 9))
        return torch.zeros((bsz, N, 6))
    pos_prop = prop_states[:,:,:3]
    vel_prop = prop_states[:,:,7:]
    q_prop = prop_states[:,:,3:7]
    hessian_rot_t = hessian_rot_t.cuda()
    hessian_state_t = hessian_state_t.cuda()
    def res_reg(states):
        position = states[:,:,:3]
        rotation = states[:,:,3:7]
        velocities = states[:,:,7:]
        Gq = attitude_jacobian(states[:,:,3:7]).detach()
        Gqprop = attitude_jacobian(prop_states[:,:,3:7]).detach()
        res_reg_state = (hessian_state_t*torch.cat([(pos_prop - position), (vel_prop-velocities)*vel_coeff], 2).unsqueeze(-2)).sum(dim=-1)
        res_reg_rot = quat_coeff*(1 - torch.abs(q_prop[0,:,None,:]@Gqprop[0]@hessian_rot_t[0]@Gq[0].transpose(-1,-2)@rotation[0,...,None]))[None, :, :, 0]
        res_reg_out = torch.cat([res_reg_state, res_reg_rot], dim=-1)
        return res_reg_out
    def res_reg_quat_only(states):
        rotation = states[:,:,3:7]
        q_prop = prop_states[:,:,3:7]
        Gq = attitude_jacobian(states[:,:,3:7]).detach()
        Gqprop = attitude_jacobian(prop_states[:,:,3:7]).detach()
        res_reg_rot = quat_coeff*(1 - torch.abs(q_prop[0,:,None,:]@Gqprop[0]@hessian_rot_t[0]@Gq[0].transpose(-1,-2)@rotation[0,...,None]))[None, :, :, 0]
        return res_reg_rot
        # return quat_coeff*(1 - torch.abs((q_prop*rotation).sum(dim=-1)).unsqueeze(-1))
    def res_reg_sum(states):
        states = states.reshape(bsz, -1, 10)
        return res_reg(states).sum(dim=0)[:,:-1].reshape(-1)
    def res_reg_sum_quat(states):
        states = states.reshape(bsz, -1, 10)
        return res_reg(states).sum(dim=0)[:,-1].reshape(-1)
    def res_reg_sum_grad(states):
        states = states.reshape(bsz, -1, 10)
        Gq = attitude_jacobian(states[:,:,3:7])
        res_out = res_reg_quat_only(states).sum()#, quat_out_only=True
        res_grad = torch.autograd.grad(res_out, states, create_graph=True)[0]
        res_grad = torch.cat([res_grad[:, :, :3], (res_grad[:, :, 3:7, None]*Gq).sum(dim=2), res_grad[:, :, 7:]], dim=2)
        return res_grad.reshape(-1)
    with torch.no_grad():
        res_reg_out = res_reg(states.cuda())#, jacobian)
        res_reg_out = res_reg_out.cpu()
    # print("finished dynamics propagation")
    if jacobian:
        Gq = attitude_jacobian(states[:,:,3:7])
        Jf = torch.autograd.functional.jacobian(res_reg_sum, states.reshape(bsz, -1).cuda(), vectorize=True).reshape(bsz, -1, 10).cpu()
        GqJ = Gq[:, None].repeat(bsz, num_res, 1, 1, 1).reshape(bsz, -1, 4, 3)
        Jf = torch.cat([Jf[:,:,:3], (Jf[:,:,3:7,None] * GqJ).sum(dim=2), Jf[:,:,7:]], dim=2)
        Jf = Jf.reshape(bsz, num_res, N*9)
        # print("finished jacobian computation")
        if GN_quat:
            Jf_quat = torch.autograd.functional.jacobian(res_reg_sum_quat, states.reshape(bsz, -1), vectorize=True).reshape(bsz, -1, 7)
            GqJ = Gq[:, None].repeat(bsz, N, 1, 1, 1).reshape(bsz, -1, 4, 3)
            Jf_quat = torch.cat([Jf_quat[:,:,:3], (Jf_quat[:,:,3:,None] * GqJ).sum(dim=2)], dim=2)
            Jf_quat = Jf_quat.reshape(bsz, N, N*6)
            qgrad = (Jf_quat*res_reg_out[:, :, -1:]).sum(dim=1).reshape(bsz, N, 6)
            Hq_full = torch.bmm(Jf_quat.transpose(-1, -2), Jf_quat)
        else:
            states.requires_grad_(True)
            qgrad = res_reg_sum_grad(states.cuda()).reshape(bsz, N, 9).cpu()
            Hdiff = torch.autograd.functional.jacobian(res_reg_sum_grad, states[:, :, :].reshape(bsz, -1).cuda(), vectorize=True).reshape(bsz, N, 9, N, 10).cpu()
            Hq_full = torch.cat([Hdiff[..., :3], (Hdiff[..., 3:7, None]*Gq[:,None,None]).sum(dim=-2), Hdiff[..., 7:]], dim=-1).reshape(bsz, N, 9, N, 9)
            Hq_full = Hq_full.reshape(bsz, N*9, N*9)
        # print("finished hessian computation")
        return res_reg_out, Jf, Hq_full, qgrad
    return res_reg_out

def predict_orbit(states, imu_meas, times, quat_coeff, vel_coeff, dt=1, jacobian=True):
    w, a = imu_meas[..., :3], imu_meas[..., 3:]
    bsz = states.shape[0]
    N = states.shape[1]
    num_res = (N-1)*6
    def res_preds(states, jac=False):
        position = states[:,:,:3]
        velocities = states[:,:,3:]
        # pos_pred, vel_pred = propagate_dynamics(position, velocities, times, dt)
        # vel = velocities + dt * a#apply_pose_transformation_quat(a, rotation) # SO3(rotation).act(a.unsqueeze(-1)).squeeze(-1)
        # pos_pred = position + dt * velocities.sum(dim=-2)
        pos_pred, vel_pred = propagate_orbit_dynamics(position, velocities, times, dt)
        # q_pred, jac_qpred = propagate_rotation_dynamics(rotation, w, times, dt, jac)
        jac_ppred = torch.eye(3,3)[None, None].repeat(bsz, N, 1, 1)
        state_pred = torch.cat([pos_pred, vel_pred], 2) 
        # vel_pred = vel
        res_pred = torch.cat([(pos_pred[:,:-1] - position[:,1:]), (vel_pred[:,:-1]-velocities[:,1:])*vel_coeff], 2)
        # print(res_pred[:, :, -1].abs().mean(), (q_pred[0,:-1]*states[0,1:,3:]).sum(dim=-1).mean(), (q_pred[0,:-1]*rotation[0,1:]).sum(dim=-1).mean())
        return res_pred, state_pred, vel_pred
    def res_preds_sum(states):
        states = states.reshape(bsz, -1, 6)
        return res_preds(states)[0].sum(dim=0).reshape(-1)
    res_pred, pose_pred, vel_pred = res_preds(states, jacobian)
    if jacobian:
        # print(res_pred[:, :, -1].abs().mean(), (pose_pred[0,:-1,3:]*states[0,1:,3:]).sum(dim=-1).mean())
        # Gq = attitude_jacobian(states[:,:,3:7])
        Jf = torch.autograd.functional.jacobian(res_preds_sum, states.reshape(bsz, -1), vectorize=True).reshape(bsz, -1, 6)
        # GqJ = Gq[:, None].repeat(bsz, num_res, 1, 1, 1).reshape(bsz, -1, 4, 3)
        # Jf = torch.cat([Jf[:,:,:3], (Jf[:,:,3:7,None] * GqJ).sum(dim=2), Jf[:,:,7:]], dim=2)
        # ipdb.set_trace()
        Jf = Jf.reshape(bsz, num_res, N*6)
        return res_pred, pose_pred, vel_pred, 0, 0, Jf    
    return res_pred, pose_pred, vel_pred


def predict_attitude(poses, velocities, imu_meas, times, quat_coeff, dt=1, jacobian=True):
    w, a = imu_meas[..., :3], imu_meas[..., 3:]
    bsz = poses.shape[0]
    N = poses.shape[1]
    num_res = (N-1)*1
    def res_preds(poses, jac=False):
        # position = poses[:,:,:3]
        rotation = poses#[:,:,3:]
        q_pred, jac_qpred = propagate_rotation_dynamics(rotation, w, times, dt, jac)
        jac_ppred = torch.eye(3,3)[None, None].repeat(bsz, N, 1, 1)
        res_pred = quat_coeff*(1 - 1*torch.abs(q_pred[:,:-1]*rotation[:,1:]).sum(dim=-1).unsqueeze(-1))
        qthat = q_pred[:,:-1]
        qt = rotation[:,1:]
        qt1 = rotation[:,:-1]
        return res_pred, quat_coeff*torch.abs(q_pred[:,:-1]*rotation[:,1:]).sum(dim=-1), jac_ppred, jac_qpred, qt, qt1, qthat
    def res_preds_sum(poses):
        poses = poses.reshape(bsz, -1, 7)
        return res_preds(poses)[0].sum(dim=0).reshape(-1)
    # def res_preds_sum_grad(poses):
    #     poses = poses.reshape(bsz, -1, 7)
    #     Gq = attitude_jacobian(poses[:,:,3:])
    #     res_out = res_preds(poses)[0][:,:,-1].sum()
    #     res_grad = torch.autograd.grad(res_out, poses, create_graph=True)[0]
    #     res_grad = torch.cat([res_grad[:, :, :3], (res_grad[:, :, 3:, None]*Gq).sum(dim=2)], dim=2)
    #     return res_grad.reshape(-1)
    def res_preds_sum_grad(poses):
        poses = poses.reshape(bsz, -1, 4)
        Gq = attitude_jacobian(poses)
        res_out = res_preds(poses)[0][:,:,-1].sum()
        res_grad = torch.autograd.grad(res_out, poses, create_graph=True)[0]
        res_grad = (res_grad[:, :, :, None]*Gq).sum(dim=2)
        # res_grad = torch.cat([res_grad[:, :, :3], (res_grad[:, :, 3:, None]*Gq).sum(dim=2)], dim=2)
        return res_grad.reshape(-1)
    res_pred, qdot, jac_ppred, jac_qpred, qt, qt1, qthat = res_preds(poses[:, :, 3:], jacobian)
    if jacobian:
        Gq = attitude_jacobian(poses[:,:,3:])
        # poses.requires_grad_(True)
        # qgrad = res_preds_sum_grad(poses).reshape(bsz, N, 6)
        # Hdiff = torch.autograd.functional.jacobian(res_preds_sum_grad, poses[:, :, :].reshape(bsz, -1), vectorize=True).reshape(bsz, N, 6, N, 7)
        # Hq_full = Hdiff = torch.cat([Hdiff[..., :3], (Hdiff[..., 3:, None]*Gq[:,None,None]).sum(dim=-2)], dim=-1).reshape(bsz, N, 6, N, 6)
        # Hq_full = Hq_full.reshape(bsz, N*6, N*6)
        # return res_pred, Hq_full, qgrad   
        if False:
            Jf = torch.autograd.functional.jacobian(res_preds_sum, poses.reshape(bsz, -1), vectorize=True).reshape(bsz, -1, 7)
            GqJ = Gq[:, None].repeat(bsz, num_res, 1, 1, 1).reshape(bsz, -1, 4, 3)
            Jf = torch.cat([Jf[:,:,:3], (Jf[:,:,3:,None] * GqJ).sum(dim=2)], dim=2)
            Jf = Jf.reshape(bsz, num_res, N*6)
            qgrad = (Jf*res_pred).sum(dim=1).reshape(bsz, N, 6)
            Hq_full = torch.bmm(Jf.transpose(-1, -2), Jf)
        else:
            Gq = attitude_jacobian(poses[:,:,3:])
            poses = poses[:, :, 3:]
            poses.requires_grad_(True)
            qgrad = res_preds_sum_grad(poses).reshape(bsz, N, 3)
            Hdiff = torch.autograd.functional.jacobian(res_preds_sum_grad, poses.reshape(bsz, -1), vectorize=True).reshape(bsz, N, 3, N, 4)
            # ipdb.set_trace()
            Hq_full = (Hdiff[..., :, None]*Gq[:,None,None]).sum(dim=-2).reshape(bsz, N, 3, N, 3)
            Hq_full = Hq_full.reshape(bsz, N*3, N*3)
        return res_pred, Hq_full, qgrad
    
    return res_pred


def get_r_sun_moon_PN(r_suns, r_moons, PNs, h, t):
    idx = ((2*t)//h).long()
    r_sun = r_suns[idx]
    r_moon = r_moons[idx]
    PN = PNs[idx]
    return r_sun, r_moon, PN

#Some constants
RE = 6378.0 #Radius of the Earth (km)
μ = 398600.0 #Standard gravitational parameter of Earth


def ground_truth_sat_dynamics(x, t, params):
    
    r = x[:,:3] #satellite position in inertial frame
    v = x[:,3:] #satellite velocity in inertial frame

    r_moons, r_suns, PNs, h = params

    r_sun, r_moon, PN = get_r_sun_moon_PN(r_suns, r_moons, PNs, h, t)
        
    # #look up this term. seems to give a rotation matrix
    # PN = bias_precession_nutation(epc)
    
    # #Compute the sun and moon positions in ECI frame
    # r_sun = sun_position(epc)
    # r_moon = moon_position(epc)
    
    #define the acceleration variable
    a = torch.zeros((x.shape[0],3)).to(x)
    
    #compute acceleration caused by Earth gravity (includes J2)
    #modeled by a spherical harmonic gravity field
    #look up this term. seems to give a rotation matrix
    # PN = bias_precession_nutation(epc)
    # Earth_r    = earth_rotation(epc)
    # rpm  = polar_motion(epc) 

    # R = rpm*Earth_r*PN
    # n_grav = 10
    # m_grav = 10
    # #main contribution in acceleration (seemed to not be equal to the Series Expansion of gravity)
    # a+= accel_gravity(x, R, n_grav, m_grav)
    
    
    #this is the gravity code that is working
    ###########################################################################################################
    #compute the gravitational acceleration based off the series expansion up to J2
    mu = 3.986004418e14 #m3/s2
    J2 = 1.08264e-3 
        
    a_2bp = (-mu*r)/(r.norm(dim=-1)**3).unsqueeze(-1)
    
    Iz = torch.tensor([0,0,1]).to(x).unsqueeze(0)
    
    a_J2 = ((3*mu*J2*R_EARTH**2)/(2*(r.norm(dim=-1)**5))).unsqueeze(-1)*((((5*((r*Iz).sum(dim=-1)**2))/r.norm(dim=-1)**2)-1).unsqueeze(-1)*r - 2*(r*Iz).sum(dim=-1).unsqueeze(-1)*Iz)     

    a_grav = a_2bp + a_J2
    
    a += a_grav
    ############################################################################################################
    
    #atmospheric drag
    #compute the atmospheric density from density harris priester model
    rho = density_harris_priester(r, r_sun)
    #ρ = 1.15e-12 #fixed atmospheric density in kg/m3

    
    #computes acceleration due to drag in inertial directions
    cd = 2.0 #drag coefficient
    area_drag = 0.1 #in m2 #area normal to the velocity direction
    m = 1.0
    
    
    a_drag = accel_drag(x, rho, m, area_drag, cd, PN)

    a += a_drag  #accel_drag(x, rho, m, area_drag, cd, PN)
    
    #Solar Radiation Pressure
    area_srp = 1.0
    coef_srp = 1.8
    a_srp = accel_srp(x, r_sun, m, area_srp, coef_srp)
    a += a_srp #accel_srp(x, r_sun, m, area_srp, coef_srp)
    
    
    #acceleration due to external bodies
    a_sun = accel_thirdbody_sun(x, r_sun)
    a+= a_sun#accel_thirdbody_sun(x, r_sun)
    
    #COMMENTED FOR TESTING
    a_moon = accel_thirdbody_moon(x, r_moon)
    a+= a_moon #accel_thirdbody_moon(x, r_moon)
    
    a_unmodeled = a_srp + a_sun + a_moon
            
    xdot = x[:,3:6]
    vdot = a
    
    x_dot = torch.cat([xdot, vdot], dim=-1)
    
    # return x_dot, a_unmodeled, rho, 
    return x_dot, a


rho_max = 5e-11 #in kg/m3
rho_min = 2e-14 #in kg/m3

def orbit_dynamics(x_orbit, t=0, params=None, mu=398600.4418, J2=1.75553e10):
    r = x_orbit[..., :3]
    v = x_orbit[..., 3:6]

    dims = len(r.shape)
    r_mat = torch.tensor(np.array([
        [6, -1.5, -1.5],
        [6, -1.5, -1.5],
        [3, -4.5, -4.5]
    ])).to(r)
    for i in range(dims-1):
        r_mat = r_mat.unsqueeze(0)

    # v_dot = -(mu / np.linalg.norm(r)**3) * r + (J2 / np.linalg.norm(r)**7) * np.dot(r, r_mat)
    v_dot = -(mu / r.norm(dim=-1, keepdim=True)**3) * r + (J2 / torch.norm(r, dim=-1, keepdim=True)**7) * (r_mat*(r[..., None, :]**2)).sum(dim=-1) * r

    return torch.cat([v, v_dot], dim=-1), v_dot

def RK4(x, t, h, params=None):
    
    # dynamics = ground_truth_sat_dynamics
    dynamics = orbit_dynamics
    f1, _ = dynamics(x, t, params) 
    f2, _ = dynamics(x+0.5*h*f1, t+h/2, params)
    f3, _ = dynamics(x+0.5*h*f2, t+h/2, params)
    f4, _ = dynamics(x+h*f3, t+h, params)
    
    xnext = x+(h/6.0)*(f1+2*f2+2*f3+f4)
        
    return xnext
     
def RK4_avg(x, t, h, params):
    f1, _ = ground_truth_sat_dynamics(x, t, params) 
    f2, _ = ground_truth_sat_dynamics(x+0.5*h*f1, t+h/2, params)
    f3, _ = ground_truth_sat_dynamics(x+0.5*h*f2, t+h/2, params)
    f4, _ = ground_truth_sat_dynamics(x+h*f3, t+h, params)
    
    favg = (1/6.0)*(f1+2*f2+2*f3+f4)
        
    return favg[:,3:6]
     

def get_all_r_sun_moon_PN():
    h = 1 #1 Hz the timestep
    # from julia import Main
    # import julia
    # jl = julia.Julia(compile_modules=False)
    # Main.include("BA/julia_utils.jl")
    # # initial time for sim
    # # epc0 = Epoch(2012, 11, 8, 12, 0, 0, 0.0)
    # # r_moons, r_suns, PNs = jl.eval("get_r_moon_sun_PNs()")
    # r_moons, r_suns, PNs = Main.get_r_moon_sun_PNs()
    # r_moons, r_suns, PNs = r_moons.transpose(), r_suns.transpose(), PNs.transpose(2,0,1)

    # np.save("data/r_moons.npy", r_moons)
    # np.save("data/r_suns.npy", r_suns)
    # np.save("data/PNs.npy", PNs)
    r_moons = np.load("data/r_moons.npy")
    r_suns = np.load("data/r_suns.npy")
    PNs = np.load("data/PNs.npy")
    ipdb.set_trace()
    r_moons, r_suns, PNs = torch.tensor(r_moons), torch.tensor(r_suns), torch.tensor(PNs)
    t = torch.arange(0, len(r_moons)/2)
    params = (r_moons, r_suns, PNs, h, t)
    return params

def quaternion_log(q):
    """
    Calculate the logarithm of a quaternion.
    
    Args:
        q: Input quaternion [x, y, z, w].
        
    Returns:
        d_theta : The quaternion logarithm
    """

    # q = (q/np.linalg.norm(q)).clamp(-1,1)
    q = (q/q.norm(dim=-1).unsqueeze(-1)).clamp(-1,1)
    # norm = q[:, :-1].norm(dim=-1)
    # theta = 2 * torch.atan2(norm, q[:, -1])
    # return q[:, :-1]*(theta/norm).unsqueeze(-1)
    theta = 2 * torch.arccos(q[..., -1])
    n = q[..., :-1] / torch.sin(theta/2).unsqueeze(-1)
    return n * theta.unsqueeze(-1)


def quaternion_exp(d_theta):
    """
    Calculate the exponential of a small change in angle (d_theta).
    
    Args:
        d_theta (np.ndarray): The small change in angle [d_theta_x, d_theta_y, d_theta_z].
        
    Returns:
        np.ndarray: The resulting quaternion [w, x, y, z].
    """
    theta = d_theta.norm(dim=-1).unsqueeze(-1)
    mask = (theta < 1e-16).double()
    Identity = torch.cat([torch.zeros_like(d_theta), torch.ones_like(theta)], dim=-1)  # Identity quaternion when theta is close to zero.
    q = torch.cat([d_theta * torch.sin(theta / 2) / (theta + 1e-16), torch.cos(theta / 2)], dim=-1)
    q = Identity*mask + q*(1-mask)
    return q

def quaternion_conjugate(q):
    """Compute the conjugate of a quaternion."""
    # return np.array([q[0], -q[1], -q[2], -q[3]])
    return torch.cat([-q[..., :-1], q[..., -1:]], dim=-1)

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    w_mul = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_mul = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_mul = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_mul = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x_mul, y_mul, z_mul, w_mul], dim=-1)

def quaternion_jacobian(q2):
    """Compute jacobian of quaternion multiplication w.r.t q1. Returns a 4x4 matrix"""
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack([
        torch.stack([w2, z2, -y2, x2], dim=-1),
        torch.stack([-z2, w2, x2, y2], dim=-1),
        torch.stack([y2, -x2, w2, z2], dim=-1),
        torch.stack([-x2, -y2, -z2, w2], dim=-1),
    ], dim=-2)

def quaternion_multiply_np(q1, q2):
    """Multiply two quaternions."""
    x1, y1, z1, w1 = np.split(q1, q1.shape[-1], axis=-1)
    x2, y2, z2, w2 = np.split(q2, q2.shape[-1], axis=-1)
    w_mul = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_mul = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_mul = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_mul = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.concatenate([x_mul, y_mul, z_mul, w_mul], axis=-1)

def apply_pose_transformation_quat(point, rotation_quaternion, translation=None):
    """Apply pose transformation (translation and quaternion rotation) on a point."""
    # Normalize the quaternion
    q_norm = rotation_quaternion/rotation_quaternion.norm(dim=-1).unsqueeze(-1)

    # Convert the point to a quaternion representation (w=0)
    v = torch.cat([point, torch.zeros_like(point[:,:,:1])], dim=-1)

    # Compute the quaternion conjugate
    q_conj = quaternion_conjugate(q_norm)

    # Apply the quaternion rotation using quaternion multiplication
    v_rotated = quaternion_multiply(q_norm, quaternion_multiply(v, q_conj))

    # Extract the transformed point coordinates
    if translation is not None:
        v_final = v_rotated[:,:,:-1] +  translation
    else:
        v_final = v_rotated[:,:,:-1]

    return v_final

def apply_pose_transformation_quat_np(point, rotation_quaternion, translation=None):
    point = torch.tensor(point).unsqueeze(0)
    rotation_quaternion = torch.tensor(rotation_quaternion).unsqueeze(0)
    if translation is not None:
        translation = torch.tensor(translation).unsqueeze(0)
    point_transformed = apply_pose_transformation_quat(point, rotation_quaternion, translation)
    return point_transformed.squeeze(0).numpy()

def apply_inverse_pose_transformation(point, rotation_quaternion, translation=None):
    """Apply pose transformation (translation and quaternion rotation) on a point."""

    if translation is not None:
        point = point - translation
    # Normalize the quaternion
    q_norm = rotation_quaternion/rotation_quaternion.norm(dim=-1).unsqueeze(-1)

    # Convert the point to a quaternion representation (w=0)
    v = torch.cat([point, torch.zeros_like(point[:,:,:1])], dim=-1)

    # Compute the quaternion conjugate
    q_conj = quaternion_conjugate(q_norm)

    # Apply the quaternion rotation using quaternion multiplication
    v_rotated = quaternion_multiply(q_conj, quaternion_multiply(v, q_norm))

    return v_rotated


#Implement this function in spherical coordinates
def gravitational_potential_new(s):
    # input: position in spherical coordinates 
    # s = [r, θ, ϕ]
    # output: gravitational potential
    
    #J2 = mu (in km) * radius of Earth^2 (km2)* J2 term
    #Constants
    mu = 3.986004418e5 #km3/s2
    J2 = 1.08264e-3 
    
    # unpack input
    r = s[:, 0]
    theta = s[:, 1]
    
    m = 1.0 #added in
    
    #only a function of the latitude
    U = (mu/r)*(1+((J2*R_EARTH**2)/(2*(r**2)))*(1-3*(torch.sin(theta))**2))
    
    return U.sum()

# conversion from cartesian coordinates to spherical coordinates
def cartesian_to_spherical(x):
    r = x[:,:3].norm(dim=-1) #torch.sqrt(x[1:3]'*x[1:3])
    theta = torch.atan2(x[:, 2],x[:, :2].norm(dim=-1)) #torch.sqrt(x[1:2]'*x[1:2]))
    phi = torch.atan2(x[:,1],x[:,0])
    
    return torch.stack([r, theta, phi], dim=-1)

def gravitational_acceleration(x):
    # input: position in cartesian coordiantes 
    # output: acceleration in cartesian coordiantes 
    
    
    q_c = x[:,:3]
    #q_d = x[7:9]
    
    v_c = x[:,3:6]
    #v_d = x[10:12]
    
    #a_d = x[7:9]
    
    #c_d = 2.0 #drag coefficient (dimensionless)
    
    #A = 0.1 
    
    #rotation of the earth (rad/s)
    #ω_earth = [0,0, OMEGA_EARTH]
    
    #v_rel = v - cross(ω_earth, q)
    
    #f_drag = -0.5*c_d*(A)*ρ*norm(v_rel)*v_rel
    
    # a_c = (ForwardDiff.gradient(_x -> gravitational_potential_new(cartesian_to_spherical(_x)), q_c))#+ a_d
    q_c.requires_grad_(True)
    a_c = torch.autograd.grad(gravitational_potential_new(cartesian_to_spherical(q_c)), q_c, create_graph=True)[0]
    #a_d = (ForwardDiff.gradient(_x -> gravitational_potential_new(cartesian_to_spherical(_x)), q_d))+ a_d
    
    return a_c

# def orbit_dynamics(x):
    
#     q_c = x[:, :3]
#     #q_d = x[7:9]
    
#     v_c = x[:, 3:6]
#     #v_d = x[10:12]
    
#     a = gravitational_acceleration(x) #obtain the gravitational acceleration given the position q
    
#     x_dot = torch.cat([v_c, a[:,:3]], dim=-1)#; zeros(3)] #x dot is velocity and acceleration
    
#     return x_dot


def RK4_satellite_potential(x,h):

    #h = 1.0 #time step
    f1 = orbit_dynamics(x)
    f2 = orbit_dynamics(x+0.5*h*f1)
    f3 = orbit_dynamics(x+0.5*h*f2)
    f4 = orbit_dynamics(x+h*f3)
    xnext = x+(h/6.0)*(f1+2*f2+2*f3+f4)
    return xnext
    
     
def RK4_orbit_dynamics_avg(x, h):
    f1 = orbit_dynamics(x) 
    f2 = orbit_dynamics(x+0.5*h*f1)
    f3 = orbit_dynamics(x+0.5*h*f2)
    f4 = orbit_dynamics(x+h*f3)
    
    favg = (1/6.0)*(f1+2*f2+2*f3+f4)
        
    return favg[:,3:6]
     


# gmst_deg = 100.0
theta_G0_deg = 280.16  # GMST at J2000.0 epoch in degrees
omega_earth_deg_per_sec = 360 / 86164.100352  # Earth's average rotational velocity in degrees per second



# Constants for the Earth's shape
a = 6378.137  # Earth's equatorial radius in kilometers
b = 6356.752  # Earth's polar radius in kilometers
e = np.sqrt(1 - (b**2 / a**2))

def deg_to_rad(deg):
    return np.deg2rad(deg)

def ecef_to_eci(x_ecef, y_ecef, z_ecef, gmst=None, times=None):
    if times is not None:
        gmst = theta_G0_deg + omega_earth_deg_per_sec * times
    # Step 1: Convert ECEF to ECI coordinates
    theta = deg_to_rad(gmst) #+ deg_to_rad(90.0)  # Convert GMST to radians and add 90 degrees

    x_eci = x_ecef * np.cos(theta) - y_ecef * np.sin(theta)
    y_eci = x_ecef * np.sin(theta) + y_ecef * np.cos(theta)
    z_eci = z_ecef

    return x_eci, y_eci, z_eci

def get_Rz(times):
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

def eci_to_ecef(r_eci, times):

    # Rotation matrix
    Rz = get_Rz(times)

    # Convert ECI to ECEF
    r_ecef = (Rz * r_eci[:, None, :]).sum(axis=-1)

    return r_ecef
    # return x_eci, y_eci, z_eci

def geodetic_to_ecef(latitude, longitude, altitude):
    # Step 1: Convert latitude and longitude to geocentric latitude
    phi = deg_to_rad(latitude)
    lambda_ = deg_to_rad(longitude)

    # Step 2: Calculate the Earth's radius of curvature in the prime vertical
    N = a / np.sqrt(1 - (e**2 * np.sin(phi)**2))
    # N = a 

    # Step 3: Convert latitude, longitude, altitude to ECEF coordinates
    x_ecef = (N + altitude) * np.cos(phi) * np.cos(lambda_)
    y_ecef = (N + altitude) * np.cos(phi) * np.sin(lambda_)
    z_ecef = ((b**2 / a**2) * N + altitude) * np.sin(phi)
    # z_ecef = (N + altitude) * np.sin(phi)

    return x_ecef, y_ecef, z_ecef

def convert_latlong_to_cartesian(lat, long, times, altitude=None):
    if altitude is None:
        altitude = np.zeros(lat.shape[0])

    # ipdb.set_trace()
    latitude, longitude = lat, long
    # Step 1: Convert latitude, longitude, altitude to ECEF coordinates
    x_ecef, y_ecef, z_ecef = geodetic_to_ecef(latitude, longitude, altitude)

    gmst_deg = theta_G0_deg + omega_earth_deg_per_sec * times
    # Step 2: Convert ECEF to ECI coordinates
    x_eci, y_eci, z_eci = ecef_to_eci(x_ecef, y_ecef, z_ecef, gmst_deg)

    return np.stack([x_eci, y_eci, z_eci], axis=-1)
    # return np.stack([x_ecef, y_ecef, z_ecef], axis=-1)


def unit_vector_to_quaternion(unit_vector1, unit_vector2):
    # Step 1: Calculate the rotation axis
    rotation_axis = np.cross(unit_vector1, unit_vector2, axis=-1)
    rotation_axis /= np.linalg.norm(rotation_axis, axis=-1)[..., None]
    
    # Step 2: Calculate the rotation angle
    rotation_angle = np.arccos(np.sum(unit_vector1*unit_vector2, axis=-1))

    # Step 3: Create the quaternion
    half_rotation_angle = 0.5 * rotation_angle
    sin_half_angle = np.sin(half_rotation_angle)

    quaternion = np.stack([
        sin_half_angle * rotation_axis[:, 0],
        sin_half_angle * rotation_axis[:,1],
        sin_half_angle * rotation_axis[:,2],
        np.cos(half_rotation_angle)
    ], axis=-1)

    return quaternion

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
    R = np.stack([xc, yc, zc], axis=-1)
    rot = transform.Rotation.from_matrix(R)
    quaternion = rot.as_quat()
    return quaternion

def convert_quaternion_to_xyz_orientation_fixed(quat, times):
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

def convert_quaternion_to_xyz_orientation(quat, times):
    # Step 1: convert quat to rotation matrix
    # NEED TO SWITCH  QUAT FROM [qw, q1, q2, q3] to [q1, q2, q3, qw]
    # quat = np.concatenate([quat[:, 1:], quat[:, :1]], axis=-1)
    rot = transform.Rotation.from_quat(quat)
    R = rot.as_matrix()

    # Step 2: comvert to ECEF from ECI
    Rz = get_Rz(times)
    R = np.matmul(Rz, R)


    # Step 3: compute the x, y, z axis
    # xc, yc, zc = R[:, :, 0], R[:, :, 1], R[:, :, 2]
    xc, yc, zc = R[:, 0], R[:, 1], R[:, 2]
    right_vector = xc
    up_vector = yc
    forward_vector = zc

    return forward_vector, up_vector, right_vector 

def convert_xyz_orientation_to_quat(xc, yc, zc, times):

    R = np.stack([-xc, -yc, zc], axis=-1)
    # right_vector = xc
    # up_vector = yc
    # forward_vector = zc

    # # Step 2: comvert from ECEF to ECI
    RzT = get_Rz(times).transpose(0, 2, 1)
    R = np.matmul(RzT, R)
    
    # Step 3: convert rotation matrix to quaternion
    rot = transform.Rotation.from_matrix(R)
    quaternion = rot.as_quat()
    # quaternion = np.concatenate([quaternion[:, 1:], quaternion[:, :1]], axis=-1)

    return quaternion


# def compute_omega_from_quat(quat, dt):
#     phis = quaternion_log(quat)
#     omega = (phis[1:] - phis[:-1]) / dt
#     omega = torch.cat([omega, torch.zeros((1, 3))], dim=0)
#     return omega

def compute_omega_from_quat(quat, dt):
    dq = quaternion_multiply(quaternion_conjugate(quat[:-1]), quat[1:])
    dq /= dq.norm(dim=-1).unsqueeze(-1)
    phi = quaternion_log(dq)
    omega = phi / dt
    omega = torch.cat([omega, torch.zeros((1, 3))], dim=0)
    return omega


def compute_velocity_from_pos(gt_pos_eci, dt):
    gt_vel_eci = (gt_pos_eci[1:] - gt_pos_eci[:-1]) / dt
    gt_vel_eci = np.concatenate([gt_vel_eci, np.zeros((1, 3))], axis=0)
    return gt_vel_eci

# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    return scatter_sum(A, ii*m + jj, dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n, mean=False):
    if mean:
        return scatter_mean(b, ii, dim=1, dim_size=n)
    return scatter_sum(b, ii, dim=1, dim_size=n)