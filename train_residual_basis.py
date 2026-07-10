import os
import warnings

warnings.filterwarnings("ignore")
import random
import sys

import d4rl
import d4rl.gym_mujoco
import gym
import numpy as np
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.append("./dppo")

from action_eigenspace import fit_action_basis, load_action_basis, save_action_basis
from env_utils import (
	ActionChunkWrapper,
	ObservationWrapperGym,
	ObservationWrapperRobomimic,
	ResidualBasisPolicyEnvWrapper,
	make_robomimic_env,
)
from residual_config import build_residual_config, parse_residual_basis_args
from utils import (
	LoggingCallback,
	collect_base_actions,
	collect_residual_basis_rollouts,
	ensure_dsrl_assets,
	init_zero_residual_weights,
	load_base_policy,
)


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
	args = parse_residual_basis_args()
	cfg = build_residual_config(args, basis=True)

	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	if cfg.use_wandb:
		wandb.init(
			project=cfg.wandb.project,
			name=cfg.name,
			group=cfg.wandb.group,
			monitor_gym=True,
			save_code=True,
			config=dict(cfg),
		)

	max_steps = int(cfg.env.max_episode_steps / cfg.act_steps)
	num_env = cfg.env.n_envs
	action_len = cfg.act_steps * cfg.action_dim

	ensure_dsrl_assets(cfg)
	base_policy = load_base_policy(cfg)
	make_env = build_make_env(cfg)

	env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
	env = ResidualBasisPolicyEnvWrapper(env, cfg, base_policy)
	env.seed(cfg.seed + 1)

	num_env_eval = cfg.env.n_eval_envs
	eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
	eval_env = ResidualBasisPolicyEnvWrapper(eval_env, cfg, base_policy)
	eval_env.seed(cfg.seed + num_env + 1)

	if cfg.train.basis_path:
		V, explained_ratio = load_action_basis(cfg.train.basis_path, n_basis=cfg.train.n_basis)
		basis_source = "file"
		print(f"Loaded basis from {cfg.train.basis_path} with shape {V.shape}")
	else:
		fit_env = make_vec_env(make_env, n_envs=1, vec_env_cls=SubprocVecEnv)
		fit_env = ResidualBasisPolicyEnvWrapper(fit_env, cfg, base_policy)
		base_actions = collect_base_actions(
			fit_env, cfg.train.basis_fit_rollouts, max_steps, cfg
		)
		fit_env.close()
		V, explained_ratio = fit_action_basis(base_actions, cfg.train.n_basis)
		basis_source = "rollout"
		basis_out = os.path.join(cfg.logdir, "action_basis.npz")
		save_action_basis(
			basis_out,
			V,
			explained_ratio,
			env_name=cfg.env_name,
			n_rollouts=cfg.train.basis_fit_rollouts,
		)
		print(f"Fitted basis from {base_actions.shape[0]} samples, saved to {basis_out}")

	if V.shape[0] != action_len:
		raise ValueError(
			f"Basis feature dim {V.shape[0]} != act_steps * action_dim ({action_len})"
		)

	env.set_basis(V)
	eval_env.set_basis(V)

	if cfg.use_wandb:
		wandb.log(
			{
				"basis/source": basis_source,
				"basis/total_explained_variance": float(explained_ratio.sum()),
				**{
					f"basis/pc_{i}_explained_variance": float(r)
					for i, r in enumerate(explained_ratio)
				},
			},
			step=0,
		)

	post_linear_modules = None
	if cfg.train.use_layer_norm:
		post_linear_modules = [torch.nn.LayerNorm]

	net_arch = [cfg.train.layer_size for _ in range(cfg.train.num_layers)]
	policy_kwargs = dict(
		net_arch=dict(pi=net_arch, qf=net_arch),
		activation_fn=torch.nn.Tanh,
		log_std_init=0.0,
		post_linear_modules=post_linear_modules,
		n_critics=cfg.train.n_critics,
	)
	model = SAC(
		"MlpPolicy",
		env,
		learning_rate=cfg.train.actor_lr,
		buffer_size=20_000_000,
		learning_starts=1,
		batch_size=cfg.train.batch_size,
		tau=cfg.train.tau,
		gamma=cfg.train.discount,
		train_freq=cfg.train.train_freq,
		gradient_steps=cfg.train.utd,
		action_noise=None,
		optimize_memory_usage=False,
		ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
		target_update_interval=1,
		target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
		use_sde=False,
		sde_sample_freq=-1,
		tensorboard_log=cfg.logdir,
		verbose=1,
		policy_kwargs=policy_kwargs,
	)
	model = init_zero_residual_weights(model)

	checkpoint_callback = CheckpointCallback(
		save_freq=cfg.save_model_interval,
		save_path=cfg.logdir + "/checkpoint/",
		name_prefix="residual_basis_policy",
		save_replay_buffer=cfg.save_replay_buffer,
		save_vecnormalize=True,
	)

	logging_callback = LoggingCallback(
		action_chunk=cfg.act_steps,
		eval_episodes=int(cfg.num_evals / num_env_eval),
		log_freq=max_steps,
		use_wandb=cfg.use_wandb,
		eval_env=eval_env,
		eval_freq=cfg.eval_interval,
		num_train_env=num_env,
		num_eval_env=num_env_eval,
		rew_offset=cfg.env.reward_offset,
		algorithm="residual_basis_sac",
		max_steps=max_steps,
		deterministic_eval=cfg.deterministic_eval,
	)

	logging_callback.evaluate(model, deterministic=False)
	if cfg.deterministic_eval:
		logging_callback.evaluate(model, deterministic=True)

	if cfg.train.init_rollout_steps > 0:
		collect_residual_basis_rollouts(
			model, env, cfg.train.init_rollout_steps, cfg
		)
		logging_callback.set_timesteps(cfg.train.init_rollout_steps * num_env)

	model.learn(
		total_timesteps=20_000_000,
		callback=[checkpoint_callback, logging_callback],
	)

	if len(cfg.name) > 0:
		model.save(cfg.logdir + "/checkpoint/final")

	env.close()
	eval_env.close()
	if cfg.use_wandb:
		wandb.finish()


if __name__ == "__main__":
	main()
