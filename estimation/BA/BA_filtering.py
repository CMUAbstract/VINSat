import torch
from BA.BA_utils import *

def BA(iter, states, velocities, imu_meas, landmarks, landmarks_xyz, ii, time_idx, intrinsics, confidences, Sigma, V, lamda_init, poses_gt_eci, initialize=False):
	states = states.double()
	v = velocities.double()
	imu_meas = imu_meas.double()
	landmarks = landmarks.double()
	landmarks_xyz = landmarks_xyz.double()
	intrinsics = intrinsics.double()
	quat_coeff = 100 #+ min(iter*10, 900)
	vel_coeff = 100#00

	bsz = states.shape[0]
	landmark_est, Jg = landmark_project(states, landmarks_xyz, intrinsics, ii, jacobian=True)
	if torch.cuda.is_available():
		r_pred, pose_pred, vel_pred, Ji, Ji_1, Jf, Hq, qgrad = predict_gpu(states, imu_meas, time_idx, quat_coeff, vel_coeff, jacobian=True, initialize=initialize) 
	else:
		r_pred, pose_pred, vel_pred, Ji, Ji_1, Jf, Hq, qgrad = predict(states, imu_meas, time_idx, quat_coeff, vel_coeff, jacobian=True, initialize=initialize)
	
	r_obs = (landmarks - landmark_est)
	alpha = min(max(1 - (2*(iter/5) - 1), 1), 2)
	c_obs = r_obs.abs().median()
	wts_obs = (((((r_obs/c_obs)**2)/abs(alpha-2) + 1)**(alpha/2 - 1)) / ((c_obs)**2)).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)[0]
	wts_obs = (wts_obs/wts_obs.max())*confidences.unsqueeze(-1).unsqueeze(-1)#*0 + 1
	Sigma = min(10000*(iter+1)**2, 1000000)
	V = 1
	dim_base = 9
	dim = 9
	Jg = Jg.reshape(bsz, -1, dim_base)[:, :, :dim].reshape(-1, 2, dim)
	
	JgTwJg = torch.bmm((Jg*wts_obs).transpose(1,2), Jg).reshape(bsz, -1, dim, dim)

	n = states.shape[1]
	ii_t = torch.tensor(ii, dtype=torch.long, device=states.device)
	JgTwJg = safe_scatter_add_vec(JgTwJg, ii_t, n).view(bsz, n, dim, dim)
	JgTwJg = torch.block_diag(*JgTwJg[0].unbind(dim=0)).unsqueeze(0)

	dim2 = min(6, dim)
	Jf = Jf.view(bsz, (n-1)*dim2, n*dim)
	JfTwJf = torch.bmm((Jf*Sigma).transpose(1,2), Jf)

	r_pred = r_pred[:, :,  :dim]
	JgT_robs = safe_scatter_add_vec((Jg.reshape(bsz, -1, 2, dim)*wts_obs[None]*r_obs.unsqueeze(-1)).sum(dim=-2), ii_t, n).view(bsz, n,dim)
	r_obs_frame = safe_scatter_add_vec(r_obs.norm(dim=-1), ii_t, n, mean=True).view(bsz, n)
	r_pred_x = r_pred[:, :, :6].clone()
	JfT_rpred = (r_pred_x.reshape(bsz, -1, 1) * Sigma * Jf).sum(dim=1).reshape(bsz, n, dim)*(-1)
	JTr = (JgT_robs + JfT_rpred - Sigma*qgrad).reshape(bsz, -1)

	lamda = lamda_init
	init_residual = torch.cat([r_obs.reshape(-1), r_pred.reshape(-1)*np.sqrt(Sigma)], dim = 0).abs().mean()
	while True:

		JTwJ =  torch.eye(n*dim)[None]*lamda+JgTwJg + JfTwJf + Sigma*Hq
		dpose = torch.linalg.solve(JTwJ, JTr).reshape(bsz, n, dim)	
		position = states[:,:,:3] + dpose[:,:,:3]
		vels = states[:,:,7:] + dpose[:,:,6:]
		rotation = quaternion_multiply(states[:,:,3:7], quaternion_exp(dpose[:,:,3:6]))
		rotation = rotation / torch.norm(rotation, dim=2, keepdim=True)
		states_new = torch.cat([position, rotation, vels], 2)
		landmark_est = landmark_project(states_new, landmarks_xyz, intrinsics, ii, jacobian=False)
		if torch.cuda.is_available():
			r_pred1, _, _ = predict_gpu(states_new, imu_meas, time_idx, quat_coeff, vel_coeff, jacobian=False, initialize=initialize) 
		else:
			r_pred1, _, _ = predict(states_new, imu_meas, time_idx, quat_coeff, vel_coeff, jacobian=False, initialize=initialize)
		r_obs1 = (landmarks - landmark_est)*wts_obs[None, :, 0]
		r_pred1 = r_pred1[:, :,  :dim].reshape(bsz, -1) * np.sqrt(Sigma)
		r_obs1 = r_obs1.reshape(bsz, -1)
		residual = torch.cat([r_obs1, r_pred1], dim = 1)
		# print("lamda: ", lamda, r_pred1.abs().mean(), r_obs1.abs().mean())

		lamda = lamda*10
		if (residual.abs().mean()) < init_residual:
			break
		if lamda > 1e4:
			print("lamda too large")
			break
		
	lamda_init = max(min(1e-1, lamda*0.01), 1e-4)

	alpha = 1
	init_residual = torch.cat([r_obs.reshape(-1), r_pred.reshape(-1)*np.sqrt(Sigma)], dim = 0).norm()
	# print("alpha: ", alpha, r_pred.abs().mean()* np.sqrt(Sigma), r_obs.abs().mean())
	# print("final quat: ", (states_new[0,:, 3:7]-poses_gt_eci[:, 3:]).abs().mean(dim=0), (1 - torch.abs((states_new[0,:, 3:7] * poses_gt_eci[:, 3:]).sum(dim=-1))).mean())
	# print("init quat: ", (states[0,:, 3:7] - poses_gt_eci[:, 3:]).abs().mean(dim=0), (1 - torch.abs((states[0,:, 3:7] * poses_gt_eci[:, 3:]).sum(dim=-1))).mean())
	if iter>18:
		print("final pos: ", (states_new[0,:, :3]-poses_gt_eci[:, :3]).abs().mean(dim=0), (states_new[0,:, :3]-poses_gt_eci[:, :3]).abs().mean(dim=0).norm().item())
	# print("init pos: ", (states[0,:, :3] - poses_gt_eci[:, :3]).abs().mean(dim=0))
	# print("final vels: ", (states_new[0,:, 7:]-velocities[0]).abs().mean(dim=0))
	# print("init vels: ", (states[0,:, 7:] - velocities[0]).abs().mean(dim=0))
	# print("r_pred :", r_pred[0,:].abs().mean(dim=0), r_pred[0,:].abs().mean(dim=0)[-1].item())
	# print("r_obs :", r_obs[0,:].abs().mean(dim=0))
	# if iter>-2:
	# 	print((states_new[0,:, :3]-poses_gt_eci[:, :3]).abs().norm(dim=-1).detach().numpy())
	# 	print("r_pred :", r_pred[0,:].abs().mean(dim=-1))
	# 	print(r_obs_frame.reshape(-1).detach().numpy())
	last_hessian = JTwJ[:, -dim:, -dim:]
	return states_new, velocities, lamda_init, last_hessian

def BA_reg(iter, states, velocities, states_prior, velocity_prior, hessian_state_t, hessian_rot_t, imu_meas, landmarks, landmarks_xyz, ii, time_idx, intrinsics, confidences, Sigma, V, lamda_init, poses_gt_eci, initialize=False, use_reg=True):
	states = states.double()
	v = velocities.double()
	imu_meas = imu_meas.double()
	landmarks = landmarks.double()
	landmarks_xyz = landmarks_xyz.double()
	intrinsics = intrinsics.double()
	quat_coeff = 100 #+ min(iter*10, 900)
	vel_coeff = 100#00
	quat_coeff_prior = 1
	vel_coeff_prior = 1

	bsz = states.shape[0]
	landmark_est, Jg = landmark_project(states, landmarks_xyz, intrinsics, ii, jacobian=True)
	if torch.cuda.is_available():
		r_pred, pose_pred, vel_pred, Ji, Ji_1, Jf, Hq, qgrad = predict_gpu(states, imu_meas, time_idx, quat_coeff, vel_coeff, jacobian=True, initialize=initialize) 
	else:
		r_pred, pose_pred, vel_pred, Ji, Ji_1, Jf, Hq, qgrad = predict(states, imu_meas, time_idx, quat_coeff, vel_coeff, jacobian=True, initialize=initialize)
	
	if torch.cuda.is_available():
		r_prior, Jp, Hqp, qgradp = prior_gpu(states, states_prior, quat_coeff_prior, vel_coeff_prior, hessian_state_t, hessian_rot_t, jacobian=True, initialize=initialize)
	else:
		r_prior, Jp, Hqp, qgradp = prior_gpu(states, states_prior, quat_coeff_prior, vel_coeff_prior, hessian_state_t, hessian_rot_t, jacobian=True, initialize=initialize)
	r_obs = (landmarks - landmark_est)
	alpha = min(max(1 - (2*(iter/5) - 1), 1), 2)
	c_obs = r_obs.abs().median()
	wts_obs = (((((r_obs/c_obs)**2)/abs(alpha-2) + 1)**(alpha/2 - 1)) / ((c_obs)**2)).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)[0]
	wts_obs = (wts_obs/wts_obs.max())*confidences.unsqueeze(-1).unsqueeze(-1)#*0 + 1
	Sigma = min(10000*(iter+1)**2, 1000000)
	V = 1
	dim_base = 9
	dim = 9
	Jg = Jg.reshape(bsz, -1, dim_base)[:, :, :dim].reshape(-1, 2, dim)
	
	JgTwJg = torch.bmm((Jg*wts_obs).transpose(1,2), Jg).reshape(bsz, -1, dim, dim)
	ipdb.set_trace()

	n = states.shape[1]
	ii_t = torch.tensor(ii, dtype=torch.long, device=states.device)
	JgTwJg = safe_scatter_add_vec(JgTwJg, ii_t, n).view(bsz, n, dim, dim)
	JgTwJg = torch.block_diag(*JgTwJg[0].unbind(dim=0)).unsqueeze(0)

	dim2 = min(6, dim)
	Jf = Jf.view(bsz, (n-1)*dim2, n*dim)
	JfTwJf = torch.bmm((Jf*Sigma).transpose(1,2), Jf)
	JpTJp = torch.bmm((Jp).transpose(1,2), Jp)

	r_pred = r_pred[:, :,  :dim]
	JgT_robs = safe_scatter_add_vec((Jg.reshape(bsz, -1, 2, dim)*wts_obs[None]*r_obs.unsqueeze(-1)).sum(dim=-2), ii_t, n).view(bsz, n,dim)
	r_obs_frame = safe_scatter_add_vec(r_obs.norm(dim=-1), ii_t, n, mean=True).view(bsz, n)
	r_pred_x = r_pred[:, :, :6].clone()
	JfT_rpred = (r_pred_x.reshape(bsz, -1, 1) * Sigma * Jf).sum(dim=1).reshape(bsz, n, dim)*(-1)

	# ipdb.set_trace()
	r_prior_x = r_prior[:, :, :6].clone()
	JpT_rprior = (r_prior_x.reshape(bsz, -1, 1) * Jp).sum(dim=1).reshape(bsz, n, dim)*(-1)
	JTr = (JgT_robs + JfT_rpred + JpT_rprior - Sigma*qgrad - qgradp).reshape(bsz, -1)
	# JTr = (JgT_robs + JfT_rpred - Sigma*qgrad).reshape(bsz, -1)

	lamda = lamda_init
	init_residual = torch.cat([r_obs.reshape(-1), r_pred.reshape(-1)*np.sqrt(Sigma), r_prior.reshape(-1)], dim = 0).abs().mean()
	while True:

		JTwJ =  torch.eye(n*dim)[None]*lamda+JgTwJg + JfTwJf + Sigma*Hq + Hqp + JpTJp
		dpose = torch.linalg.solve(JTwJ, JTr).reshape(bsz, n, dim)	
		position = states[:,:,:3] + dpose[:,:,:3]
		vels = states[:,:,7:] + dpose[:,:,6:]
		rotation = quaternion_multiply(states[:,:,3:7], quaternion_exp(dpose[:,:,3:6]))
		rotation = rotation / torch.norm(rotation, dim=2, keepdim=True)
		states_new = torch.cat([position, rotation, vels], 2)
		landmark_est = landmark_project(states_new, landmarks_xyz, intrinsics, ii, jacobian=False)
		if torch.cuda.is_available():
			r_pred1, _, _ = predict_gpu(states_new, imu_meas, time_idx, quat_coeff_prior, vel_coeff, jacobian=False, initialize=initialize)
			r_prior1 = prior_gpu(states_new, states_prior, quat_coeff_prior, vel_coeff, hessian_state_t, hessian_rot_t, jacobian=False, initialize=initialize) 
		else:
			r_pred1, _, _ = predict(states_new, imu_meas, time_idx, quat_coeff_prior, vel_coeff, jacobian=False, initialize=initialize)
			r_prior1 = prior_gpu(states_new, states_prior, quat_coeff_prior, vel_coeff, hessian_state_t, hessian_rot_t, jacobian=False, initialize=initialize)
		r_obs1 = (landmarks - landmark_est)*wts_obs[None, :, 0]
		r_pred1 = r_pred1[:, :,  :dim].reshape(bsz, -1) * np.sqrt(Sigma)
		r_prior1 = r_prior1[:, :,  :dim].reshape(bsz, -1)
		r_obs1 = r_obs1.reshape(bsz, -1)
		residual = torch.cat([r_obs1, r_pred1, r_prior1], dim = 1)
		print("lamda: ", lamda, r_pred1.abs().mean(), r_obs1.abs().mean(), r_prior1.abs().mean())

		lamda = lamda*10
		if (residual.abs().mean()) < init_residual:
			break
		if lamda > 1e4:
			print("lamda too large")
			break
		
	lamda_init = max(min(1e-1, lamda*0.01), 1e-4)

	alpha = 1
	init_residual = torch.cat([r_obs.reshape(-1), r_pred.reshape(-1)*np.sqrt(Sigma)], dim = 0).norm()
	# print("alpha: ", alpha, r_pred.abs().mean()* np.sqrt(Sigma), r_obs.abs().mean())
	# print("final quat: ", (states_new[0,:, 3:7]-poses_gt_eci[:, 3:]).abs().mean(dim=0), (1 - torch.abs((states_new[0,:, 3:7] * poses_gt_eci[:, 3:]).sum(dim=-1))).mean())
	# print("init quat: ", (states[0,:, 3:7] - poses_gt_eci[:, 3:]).abs().mean(dim=0), (1 - torch.abs((states[0,:, 3:7] * poses_gt_eci[:, 3:]).sum(dim=-1))).mean())
	print("final pos: ", (states_new[0,:, :3]-poses_gt_eci[:, :3]).abs().mean(dim=0), (states_new[0,:, :3]-poses_gt_eci[:, :3]).abs().mean(dim=0).norm().item())
	# print("init pos: ", (states[0,:, :3] - poses_gt_eci[:, :3]).abs().mean(dim=0))
	# print("final vels: ", (states_new[0,:, 7:]-velocities[0]).abs().mean(dim=0))
	# print("init vels: ", (states[0,:, 7:] - velocities[0]).abs().mean(dim=0))
	# print("r_pred :", r_pred[0,:].abs().mean(dim=0), r_pred[0,:].abs().mean(dim=0)[-1].item())
	# print("r_obs :", r_obs[0,:].abs().mean(dim=0))
	# if iter>-2:
	# 	print((states_new[0,:, :3]-poses_gt_eci[:, :3]).abs().norm(dim=-1).detach().numpy())
	# 	print("r_pred :", r_pred[0,:].abs().mean(dim=-1))
	# 	print(r_obs_frame.reshape(-1).detach().numpy())

	last_hessian = JTwJ[:, -dim:, -dim:]
	return states_new, velocities, lamda_init, last_hessian


