import argparse
import os
import random
import sys

import d4rl
import d4rl.gym_mujoco
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.append("./dppo")

from action_eigenspace import cumulative_explained_variance, save_action_basis
from env_utils import (
	ActionChunkWrapper,
	ObservationWrapperGym,
	ObservationWrapperRobomimic,
	ResidualPolicyEnvWrapper,
	make_robomimic_env,
)
from residual_config import add_common_residual_args, build_residual_config
from utils import collect_base_actions, ensure_dsrl_assets, load_base_policy


def build_make_env(cfg):
	def make_env():
		if cfg.env_name in [
			"halfcheetah-medium-v2",
			"hopper-medium-v2",
			"walker2d-medium-v2",
		]:
			env = gym.make(cfg.env_name)
			env = ObservationWrapperGym(env, cfg.normalization_path)
		elif cfg.env_name in ["lift", "can", "square", "transport"]:
			env = make_robomimic_env(
				env=cfg.env_name,
				normalization_path=cfg.normalization_path,
				low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
				dppo_path=cfg.dppo_path,
			)
			env = ObservationWrapperRobomimic(
				env, reward_offset=cfg.env.reward_offset
			)
		env = ActionChunkWrapper(
			env, cfg, max_episode_steps=cfg.env.max_episode_steps
		)
		return env

	return make_env


def main():
	parser = argparse.ArgumentParser(
		description="Collect base-policy rollouts and plot PCA explained variance"
	)
	add_common_residual_args(parser)
	parser.add_argument("--num-rollouts", type=int, default=20)
	parser.add_argument(
		"--output",
		type=str,
		default="./plots/action_pca_explained_variance.png",
	)
	parser.add_argument(
		"--save-basis",
		type=str,
		default="",
		help="Optional path to save fitted basis .npz",
	)
	args = parser.parse_args()

	cfg = build_residual_config(args, basis=False)
	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	max_steps = int(cfg.env.max_episode_steps / cfg.act_steps)
	ensure_dsrl_assets(cfg)
	base_policy = load_base_policy(cfg)

	env = make_vec_env(build_make_env(cfg), n_envs=1, vec_env_cls=SubprocVecEnv)
	env = ResidualPolicyEnvWrapper(env, cfg, base_policy)

	print(f"Collecting {args.num_rollouts} rollouts ({max_steps} steps each)...")
	base_actions = collect_base_actions(env, args.num_rollouts, max_steps, cfg)
	env.close()
	print(f"Collected {base_actions.shape[0]} base action vectors (dim={base_actions.shape[1]})")

	n_components, cumvar = cumulative_explained_variance(base_actions)
	action_len = cfg.act_steps * cfg.action_dim
	max_k = min(action_len, base_actions.shape[0])
	n_components = n_components[:max_k]
	cumvar = cumvar[:max_k]

	os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
	fig, ax = plt.subplots(figsize=(7, 4.5))
	ax.plot(n_components, cumvar, marker="o", markersize=4, linewidth=1.5)
	ax.set_xlabel("Number of eigenvectors")
	ax.set_ylabel("Cumulative explained variance")
	ax.set_title(
		f"PCA of base diffusion actions ({cfg.env_name}, {args.num_rollouts} rollouts)"
	)
	ax.set_xlim(1, max_k)
	ax.set_ylim(0.0, 1.02)
	ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(args.output, dpi=150)
	print(f"Saved plot to {args.output}")

	if args.save_basis:
		from action_eigenspace import fit_action_basis

		n_basis = min(cfg.train.n_basis, max_k)
		V, explained = fit_action_basis(base_actions, n_basis)
		save_action_basis(
			args.save_basis,
			V,
			explained,
			env_name=cfg.env_name,
			n_rollouts=args.num_rollouts,
		)
		print(f"Saved basis ({V.shape}) to {args.save_basis}")


if __name__ == "__main__":
	main()
