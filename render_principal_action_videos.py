"""
Render videos for principal-component residual directions on robomimic.

For each PC k, holds coefficient z = e_k fixed and plays the composed action
  a = a_base + residual_scale * (z @ V.T)
for --num-steps chunk steps, recording agentview frames.
"""

import argparse
import os
import random
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import d4rl
import d4rl.gym_mujoco
import gym
import imageio
import numpy as np
import torch

sys.path.append("./dppo")

from action_eigenspace import fit_action_basis, load_action_basis, save_action_basis
from env_utils import (
	ActionChunkWrapper,
	ObservationWrapperRobomimic,
	make_robomimic_env,
)
from residual_config import add_common_residual_args, build_residual_config
from utils import collect_base_actions, ensure_dsrl_assets, load_base_policy


def build_env_stack(cfg):
	lowdim_env = make_robomimic_env(
		render=True,
		env=cfg.env_name,
		normalization_path=cfg.normalization_path,
		low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
		dppo_path=cfg.dppo_path,
	)
	obs_env = ObservationWrapperRobomimic(
		lowdim_env, reward_offset=cfg.env.reward_offset
	)
	chunk_env = ActionChunkWrapper(
		obs_env, cfg, max_episode_steps=cfg.env.max_episode_steps
	)
	return chunk_env, lowdim_env


def collect_or_load_basis(cfg, basis_path, num_rollouts, n_basis, base_policy):
	if basis_path and os.path.isfile(basis_path):
		V, explained = load_action_basis(basis_path, n_basis=n_basis)
		print(f"Loaded basis from {basis_path} ({V.shape})")
		return V, explained

	from stable_baselines3.common.env_util import make_vec_env
	from stable_baselines3.common.vec_env import SubprocVecEnv
	from env_utils import ResidualPolicyEnvWrapper

	def make_env():
		lowdim_env = make_robomimic_env(
			render=False,
			env=cfg.env_name,
			normalization_path=cfg.normalization_path,
			low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
			dppo_path=cfg.dppo_path,
		)
		obs_env = ObservationWrapperRobomimic(
			lowdim_env, reward_offset=cfg.env.reward_offset
		)
		return ActionChunkWrapper(
			obs_env, cfg, max_episode_steps=cfg.env.max_episode_steps
		)

	max_steps = int(cfg.env.max_episode_steps / cfg.act_steps)
	vec_env = make_vec_env(make_env, n_envs=1, vec_env_cls=SubprocVecEnv)
	vec_env = ResidualPolicyEnvWrapper(vec_env, cfg, base_policy)
	base_actions = collect_base_actions(vec_env, num_rollouts, max_steps, cfg)
	vec_env.close()
	V, explained = fit_action_basis(base_actions, n_basis)
	if basis_path:
		save_action_basis(
			basis_path,
			V,
			explained,
			env_name=cfg.env_name,
			n_rollouts=num_rollouts,
		)
		print(f"Saved basis to {basis_path}")
	return V, explained


def render_pc_video(
	chunk_env,
	lowdim_env,
	base_policy,
	V,
	pc_index,
	cfg,
	video_path,
	num_steps,
	residual_scale,
	base_noise_mode,
):
	device = cfg.device
	action_len = cfg.act_steps * cfg.action_dim
	z = np.zeros(V.shape[1], dtype=np.float32)
	z[pc_index] = 1.0
	residual_flat = (z @ V.T).astype(np.float32)

	writer = imageio.get_writer(video_path, fps=30)
	obs, _ = chunk_env.reset(seed=cfg.seed)
	chunk_env.count = 0

	for _ in range(num_steps):
		obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
		if base_noise_mode == "zero":
			base_noise = torch.zeros(
				1, cfg.act_steps, cfg.action_dim, device=device
			)
		else:
			base_noise = torch.randn(
				1, cfg.act_steps, cfg.action_dim, device=device
			)
		base_action = base_policy(obs_tensor, base_noise, return_numpy=False)[0]
		residual = (
			torch.tensor(residual_flat, device=device, dtype=torch.float32)
			.reshape(cfg.act_steps, cfg.action_dim)
		)
		final_action = base_action + residual_scale * residual
		obs, reward, done, truncated, info = chunk_env.step(
			final_action.detach().cpu().numpy().reshape(action_len)
		)
		frame = lowdim_env.render(mode="rgb_array")
		writer.append_data(frame)
		if done or truncated:
			break

	writer.close()
	print(f"Saved {video_path}")


def main():
	parser = argparse.ArgumentParser(
		description="Render principal-component action videos"
	)
	add_common_residual_args(parser)
	parser.add_argument("--num-rollouts", type=int, default=20)
	parser.add_argument("--num-pcs", type=int, default=8, help="Number of PCs to render")
	parser.add_argument("--num-steps", type=int, default=10)
	parser.add_argument(
		"--basis-path",
		type=str,
		default="./plots/can_basis.npz",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="./plots/pc_videos",
	)
	args = parser.parse_args()

	cfg = build_residual_config(args, basis=False)
	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	ensure_dsrl_assets(cfg)
	base_policy = load_base_policy(cfg)
	V, explained = collect_or_load_basis(
		cfg,
		args.basis_path,
		args.num_rollouts,
		args.n_basis,
		base_policy,
	)

	action_len = cfg.act_steps * cfg.action_dim
	if V.shape[0] != action_len:
		raise ValueError(f"Basis dim {V.shape[0]} != action_len {action_len}")

	residual_scale = cfg.train.residual_scale
	base_noise_mode = cfg.train.base_noise_mode
	num_pcs = min(args.num_pcs, V.shape[1])

	os.makedirs(args.output_dir, exist_ok=True)
	chunk_env, lowdim_env = build_env_stack(cfg)

	for pc in range(num_pcs):
		video_path = os.path.join(
			args.output_dir,
			f"pc{pc + 1}_explained_{explained[pc]:.3f}.mp4",
		)
		render_pc_video(
			chunk_env,
			lowdim_env,
			base_policy,
			V,
			pc,
			cfg,
			video_path,
			args.num_steps,
			residual_scale,
			base_noise_mode,
		)

	chunk_env.close()
	print(f"Rendered {num_pcs} videos to {args.output_dir}")


if __name__ == "__main__":
	main()
