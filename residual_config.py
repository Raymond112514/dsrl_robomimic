"""Argparse-based config for residual RL training and utilities."""

from __future__ import annotations

import argparse
from datetime import datetime

from omegaconf import OmegaConf

from utils import ROBOMIMIC_RESIDUAL_TASK_DEFAULTS

LOW_DIM_KEYS = [
	"robot0_eef_pos",
	"robot0_eef_quat",
	"robot0_gripper_qpos",
	"object",
]

COND_STEPS = 1


def _timestamp() -> str:
	return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def add_common_residual_args(parser: argparse.ArgumentParser) -> None:
	parser.add_argument("--env-name", type=str, default="can", choices=["can", "lift", "square"])
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--device", type=str, default="cuda:0")
	parser.add_argument("--log-dir", type=str, default="./logs")
	parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--deterministic-eval", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--dppo-path", type=str, default="./dppo")
	parser.add_argument("--eval-interval", type=int, default=3000)
	parser.add_argument("--num-evals", type=int, default=200)
	parser.add_argument("--save-model-interval", type=int, default=10000)
	parser.add_argument("--save-replay-buffer", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--n-envs", type=int, default=4)
	parser.add_argument("--n-eval-envs", type=int, default=25)
	parser.add_argument("--max-episode-steps", type=int, default=300)
	parser.add_argument("--reward-offset", type=int, default=1)
	parser.add_argument("--actor-lr", type=float, default=3e-4)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--tau", type=float, default=0.005)
	parser.add_argument("--utd", type=int, default=20)
	parser.add_argument("--use-layer-norm", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--layer-size", type=int, default=2048)
	parser.add_argument("--num-layers", type=int, default=3)
	parser.add_argument("--discount", type=float, default=0.99)
	parser.add_argument("--ent-coef", type=float, default=-1)
	parser.add_argument("--target-ent", type=float, default=0.0)
	parser.add_argument("--init-rollout-steps", type=int, default=1501)
	parser.add_argument("--action-magnitude", type=float, default=2.0)
	parser.add_argument("--residual-scale", type=float, default=0.01)
	parser.add_argument("--base-noise-mode", type=str, default="random", choices=["random", "zero"])
	parser.add_argument("--n-critics", type=int, default=2)
	parser.add_argument("--n-basis", type=int, default=8)
	parser.add_argument("--basis-fit-rollouts", type=int, default=20)


def add_basis_args(parser: argparse.ArgumentParser) -> None:
	parser.add_argument(
		"--basis-path",
		type=str,
		default="",
		help="Optional path to a precomputed action basis .npz",
	)


def parse_residual_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train residual SAC on robomimic")
	add_common_residual_args(parser)
	return parser.parse_args(argv)


def parse_residual_basis_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train residual-basis SAC on robomimic")
	add_common_residual_args(parser)
	add_basis_args(parser)
	return parser.parse_args(argv)


def build_residual_config(args: argparse.Namespace, basis: bool = False) -> OmegaConf:
	if args.env_name not in ROBOMIMIC_RESIDUAL_TASK_DEFAULTS:
		raise ValueError(
			f"Unsupported env_name={args.env_name!r}. "
			f"Choose from {list(ROBOMIMIC_RESIDUAL_TASK_DEFAULTS)}."
		)

	task = ROBOMIMIC_RESIDUAL_TASK_DEFAULTS[args.env_name]
	ts = _timestamp()
	suffix = "residual-basis" if basis else "residual"
	name = f"robomimic_{args.env_name}_{suffix.replace('-', '_')}_{ts}_{args.seed}"
	logdir = f"{args.log_dir}/robomimic-{suffix}/{name}/{ts}_{args.seed}"

	base_obs_dim = task["base_obs_dim"]
	action_dim = task["action_dim"]
	act_steps = task["act_steps"]
	base_policy_path = task["base_policy_path"]
	normalization_path = f"./dppo/log/robomimic/{args.env_name}/normalization.npz"

	wandb_group = f"robomimic-{args.env_name}-{suffix}"

	cfg_dict = {
		"name": name,
		"logdir": logdir,
		"base_policy_path": base_policy_path,
		"dppo_path": args.dppo_path,
		"normalization_path": normalization_path,
		"seed": args.seed,
		"device": args.device,
		"use_wandb": args.use_wandb,
		"env_name": args.env_name,
		"log_dir": args.log_dir,
		"base_obs_dim": base_obs_dim,
		"obs_dim": base_obs_dim,
		"action_dim": action_dim,
		"cond_steps": COND_STEPS,
		"act_steps": act_steps,
		"deterministic_eval": args.deterministic_eval,
		"eval_interval": args.eval_interval,
		"num_evals": args.num_evals,
		"save_model_interval": args.save_model_interval,
		"save_replay_buffer": args.save_replay_buffer,
		"env": {
			"n_envs": args.n_envs,
			"n_eval_envs": args.n_eval_envs,
			"name": args.env_name,
			"max_episode_steps": args.max_episode_steps,
			"reset_at_iteration": False,
			"save_video": False,
			"best_reward_threshold_for_success": 1,
			"reward_offset": args.reward_offset,
			"wrappers": {
				"robomimic_lowdim": {
					"normalization_path": normalization_path,
					"low_dim_keys": LOW_DIM_KEYS,
				},
				"multi_step": {
					"n_obs_steps": COND_STEPS,
					"n_action_steps": act_steps,
					"max_episode_steps": args.max_episode_steps,
					"reset_within_step": True,
				},
			},
		},
		"wandb": {
			"project": "dsrl",
			"run": f"{ts.split('_', 1)[1]}_{name}",
			"group": wandb_group,
		},
		"train": {
			"tau": args.tau,
			"actor_lr": args.actor_lr,
			"batch_size": args.batch_size,
			"train_freq": 1,
			"utd": args.utd,
			"use_layer_norm": args.use_layer_norm,
			"layer_size": args.layer_size,
			"num_layers": args.num_layers,
			"discount": args.discount,
			"ent_coef": args.ent_coef,
			"target_ent": args.target_ent,
			"init_rollout_steps": args.init_rollout_steps,
			"action_magnitude": args.action_magnitude,
			"residual_scale": args.residual_scale,
			"base_noise_mode": args.base_noise_mode,
			"n_basis": args.n_basis,
			"basis_fit_rollouts": args.basis_fit_rollouts,
			"n_critics": args.n_critics,
		},
		"model": {
			"_target_": "model.diffusion.diffusion_eval.DiffusionEval",
			"ft_denoising_steps": 0,
			"predict_epsilon": True,
			"denoised_clip_value": 1.0,
			"randn_clip_value": 3,
			"network_path": base_policy_path,
			"network": {
				"_target_": "model.diffusion.mlp_diffusion.DiffusionMLP",
				"time_dim": 16,
				"mlp_dims": [512, 512, 512],
				"residual_style": True,
				"cond_dim": base_obs_dim * COND_STEPS,
				"horizon_steps": act_steps,
				"action_dim": action_dim,
			},
			"horizon_steps": act_steps,
			"obs_dim": base_obs_dim,
			"action_dim": action_dim,
			"denoising_steps": 20,
			"device": args.device,
			"use_ddim": True,
			"ddim_steps": 8,
			"controllable_noise": True,
		},
	}

	if basis:
		cfg_dict["train"]["basis_path"] = args.basis_path or None

	return OmegaConf.create(cfg_dict)
