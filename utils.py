import os
import sys

import torch
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import hydra
from omegaconf import OmegaConf


ROBOMIMIC_RESIDUAL_TASK_DEFAULTS = {
	"can": {
		"base_obs_dim": 23,
		"action_dim": 7,
		"act_steps": 4,
		"base_policy_path": (
			"./dppo/log/robomimic-pretrain/can/"
			"can_pre_diffusion_mlp_ta4_td20/2024-06-28_13-29-54/checkpoint/state_5000.pt"
		),
	},
	"lift": {
		"base_obs_dim": 19,
		"action_dim": 7,
		"act_steps": 4,
		"base_policy_path": (
			"./dppo/log/robomimic-pretrain/lift/"
			"lift_pre_diffusion_mlp_ta4_td20/2024-06-28_14-47-58/checkpoint/state_5000.pt"
		),
	},
	"square": {
		"base_obs_dim": 23,
		"action_dim": 7,
		"act_steps": 4,
		"base_policy_path": (
			"./dppo/log/robomimic-pretrain/square/"
			"square_pre_diffusion_mlp_ta4_td100_ddim-100steps/"
			"2025-04-11_19-13-26_44/checkpoint/state_3000.pt"
		),
	},
}


def apply_robomimic_residual_task_defaults(cfg, basis=False):
	"""Align checkpoint/obs dims with env_name."""
	task = cfg.env_name
	if task in cfg.base_policy_path:
		return

	defaults = ROBOMIMIC_RESIDUAL_TASK_DEFAULTS.get(task)
	if defaults is None:
		raise ValueError(
			f"Config mismatch: env_name={task!r} but base_policy_path is for a "
			f"different task ({cfg.base_policy_path}). No built-in defaults for {task!r}; "
			f"use a task-specific config yaml."
		)

	print(
		f"Auto-selecting residual defaults for {task!r} "
		f"(base_obs_dim={defaults['base_obs_dim']}, checkpoint updated)."
	)
	OmegaConf.set_struct(cfg, False)
	cfg.base_obs_dim = defaults["base_obs_dim"]
	cfg.obs_dim = defaults["base_obs_dim"]
	cfg.base_policy_path = defaults["base_policy_path"]
	cfg.action_dim = defaults["action_dim"]
	cfg.act_steps = defaults["act_steps"]
	cfg.model.obs_dim = defaults["base_obs_dim"]
	cfg.model.action_dim = defaults["action_dim"]
	cfg.model.horizon_steps = defaults["act_steps"]
	cfg.model.network.cond_dim = defaults["base_obs_dim"] * cfg.cond_steps
	cfg.model.network.horizon_steps = defaults["act_steps"]
	cfg.model.network.action_dim = defaults["action_dim"]
	suffix = "residual-basis" if basis else "residual"
	cfg.wandb.group = f"robomimic-{task}-{suffix}"
	OmegaConf.set_struct(cfg, True)


class DPPOBasePolicyWrapper:
	def __init__(self, base_policy):
		self.base_policy = base_policy
		
	def __call__(self, obs, initial_noise, return_numpy=True):
		cond = {
			"state": obs,
			"noise_action": initial_noise,
		}
		with torch.no_grad():
			samples = self.base_policy(cond=cond, deterministic=True)
		diffused_actions = (samples.trajectories.detach())
		if return_numpy:
			diffused_actions = diffused_actions.cpu().numpy()
		return diffused_actions	


def ensure_dsrl_assets(cfg):
	"""Download normalization stats and base policy checkpoint if missing."""
	import gdown

	dppo_script_path = os.path.join(os.path.dirname(__file__), "dppo", "script")
	if dppo_script_path not in sys.path:
		sys.path.append(dppo_script_path)
	from download_url import get_checkpoint_download_url, get_normalization_download_url

	if "normalization_path" in cfg and not os.path.exists(cfg.normalization_path):
		download_url = get_normalization_download_url(cfg)
		download_target = cfg.normalization_path
		os.makedirs(os.path.dirname(download_target), exist_ok=True)
		print(
			f"Downloading normalization statistics from {download_url} to {download_target}"
		)
		gdown.download(url=download_url, output=download_target, fuzzy=True)

	if "base_policy_path" in cfg and not os.path.exists(cfg.base_policy_path):
		download_url = get_checkpoint_download_url(cfg)
		if download_url is None:
			raise ValueError(
				f"Unknown checkpoint path: {cfg.base_policy_path}. "
				"Download checkpoints from https://drive.google.com/drive/folders/1kzC49RRFOE7aTnJh_7OvJ1K5XaDmtuh1 "
				"and place them in ./dppo/log."
			)
		download_target = cfg.base_policy_path
		os.makedirs(os.path.dirname(download_target), exist_ok=True)
		print(f"Downloading checkpoint from {download_url} to {download_target}")
		gdown.download(url=download_url, output=download_target, fuzzy=True)


def load_base_policy(cfg):
	base_policy = hydra.utils.instantiate(cfg.model)
	base_policy = base_policy.eval()
	return DPPOBasePolicyWrapper(base_policy)


class LoggingCallback(BaseCallback):
	def __init__(self, 
		action_chunk=4, 
		log_freq=1000,
		use_wandb=True, 
		eval_env=None, 
		eval_freq=70, 
		eval_episodes=2, 
		verbose=0, 
		rew_offset=0, 
		num_train_env=1,
		num_eval_env=1,
		algorithm='dsrl_sac',
		max_steps=-1,
		deterministic_eval=False,
	):
		super().__init__(verbose)
		self.action_chunk = action_chunk
		self.log_freq = log_freq
		self.episode_rewards = []
		self.episode_lengths = []
		self.use_wandb = use_wandb
		self.eval_env = eval_env
		self.eval_episodes = eval_episodes
		self.eval_freq = eval_freq
		self.log_count = 0
		self.total_reward = 0
		self.rew_offset = rew_offset
		self.total_timesteps = 0
		self.num_train_env = num_train_env
		self.num_eval_env = num_eval_env
		self.episode_success = np.zeros(self.num_train_env)
		self.episode_completed = np.zeros(self.num_train_env)
		self.algorithm = algorithm
		self.max_steps = max_steps
		self.deterministic_eval = deterministic_eval

	def _wandb_step(self):
		step = int(self.log_count)
		if self.use_wandb and wandb.run is not None and wandb.run.step is not None:
			step = max(step, int(wandb.run.step))
		return step

	def _commit_wandb_step(self, step):
		self.log_count = max(int(self.log_count), int(step) + 1)

	def _on_step(self):
		for info in self.locals['infos']:
			if 'episode' in info:
				self.episode_rewards.append(info['episode']['r'])
				self.episode_lengths.append(info['episode']['l'])
		rew = self.locals['rewards']
		self.total_reward += np.mean(rew)
		self.episode_success[rew > -self.rew_offset] = 1
		self.episode_completed[self.locals['dones']] = 1
		self.total_timesteps += self.action_chunk * self.model.n_envs
		if self.n_calls % self.log_freq == 0:
			if len(self.episode_rewards) > 0:
				if self.use_wandb:
					step = self._wandb_step()
					wandb.log({
						"train/ep_len_mean": np.mean(self.episode_lengths),
						"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						"train/ep_rew_mean": np.mean(self.episode_rewards),
						"train/rew_mean": np.mean(self.total_reward),
						"train/timesteps": self.total_timesteps,
						"train/ent_coef": self.locals['self'].logger.name_to_value['train/ent_coef'],
						"train/actor_loss": self.locals['self'].logger.name_to_value['train/actor_loss'],
						"train/critic_loss": self.locals['self'].logger.name_to_value['train/critic_loss'],
						"train/ent_coef_loss": self.locals['self'].logger.name_to_value['train/ent_coef_loss'],
					}, step=step)
					if np.sum(self.episode_completed) > 0:
						wandb.log({
							"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						}, step=step)
					if self.algorithm == 'dsrl_na':
						wandb.log({
							"train/noise_critic_loss": self.locals['self'].logger.name_to_value['train/noise_critic_loss'],
						}, step=step)
					self._commit_wandb_step(step)
				self.episode_rewards = []
				self.episode_lengths = []
				self.total_reward = 0
				self.episode_success = np.zeros(self.num_train_env)
				self.episode_completed = np.zeros(self.num_train_env)

		if self.n_calls % self.eval_freq == 0:
			self.evaluate(self.locals['self'], deterministic=False)
			if self.deterministic_eval:
				self.evaluate(self.locals['self'], deterministic=True)
		return True
	
	def evaluate(self, agent, deterministic=False):
		if self.eval_episodes > 0:
			env = self.eval_env
			with torch.no_grad():
				success, rews = [], []
				rew_total, total_ep = 0, 0
				rew_ep = np.zeros(self.num_eval_env)
				for i in range(self.eval_episodes):
					obs = env.reset()
					success_i = np.zeros(obs.shape[0])
					r = []
					for _ in range(self.max_steps):
						if self.algorithm in ('dsrl_sac', 'residual_sac', 'residual_basis_sac'):
							action, _ = agent.predict(obs, deterministic=deterministic)
						elif self.algorithm == 'dsrl_na':
							action, _ = agent.predict_diffused(obs, deterministic=deterministic)
						next_obs, reward, done, info = env.step(action)
						obs = next_obs
						rew_ep += reward
						rew_total += sum(rew_ep[done])
						rew_ep[done] = 0 
						total_ep += np.sum(done)
						success_i[reward > -self.rew_offset] = 1
						r.append(reward)
					success.append(success_i.mean())
					rews.append(np.mean(np.array(r)))
					print(f'eval episode {i} at timestep {self.total_timesteps}')
				success_rate = np.mean(success)
				if total_ep > 0:
					avg_rew = rew_total / total_ep
				else:
					avg_rew = 0
				if self.use_wandb:
					name = 'eval'
					step = self._wandb_step()
					if deterministic:
						wandb.log({
							f"{name}/success_rate_deterministic": success_rate,
							f"{name}/reward_deterministic": avg_rew,
						}, step=step)
					else:
						wandb.log({
							f"{name}/success_rate": success_rate,
							f"{name}/reward": avg_rew,
							f"{name}/timesteps": self.total_timesteps,
						}, step=step)
					self._commit_wandb_step(step)

	def set_timesteps(self, timesteps):
		self.total_timesteps = timesteps



def collect_rollouts(model, env, num_steps, base_policy, cfg):
	obs = env.reset()
	for i in range(num_steps):
		noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
		if cfg.algorithm == 'dsrl_sac':
			noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
			noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
		action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
		next_obs, reward, done, info = env.step(action)
		if cfg.algorithm == 'dsrl_na':
			action_store = action
		elif cfg.algorithm == 'dsrl_sac':
			action_store = noise.detach().cpu().numpy()
		action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
		if cfg.algorithm == 'dsrl_sac':
			action_store = model.policy.scale_action(action_store)
		model.replay_buffer.add(
				obs=obs,
				next_obs=next_obs,
				action=action_store,
				reward=reward,
				done=done,
				infos=info,
			)
		obs = next_obs
	model.replay_buffer.final_offline_step()


def init_zero_residual_weights(model):
	actor = model.policy.actor
	if hasattr(actor, "mu"):
		if isinstance(actor.mu, torch.nn.Linear):
			torch.nn.init.zeros_(actor.mu.weight)
			if actor.mu.bias is not None:
				torch.nn.init.zeros_(actor.mu.bias)
		elif isinstance(actor.mu, torch.nn.Sequential):
			for layer in actor.mu:
				if isinstance(layer, torch.nn.Linear):
					torch.nn.init.zeros_(layer.weight)
					if layer.bias is not None:
						torch.nn.init.zeros_(layer.bias)
	return model


def collect_residual_rollouts(model, env, num_steps, cfg):
	obs = env.reset()
	action_len = cfg.act_steps * cfg.action_dim
	zero_action = np.zeros((cfg.env.n_envs, action_len), dtype=np.float32)
	for _ in range(num_steps):
		scaled_action = model.policy.scale_action(zero_action)
		next_obs, reward, done, info = env.step(scaled_action)
		model.replay_buffer.add(
			obs=obs,
			next_obs=next_obs,
			action=scaled_action,
			reward=reward,
			done=done,
			infos=info,
		)
		obs = next_obs
	model.replay_buffer.final_offline_step()


def collect_base_actions(env, num_rollouts, max_steps, cfg):
	action_len = cfg.act_steps * cfg.action_dim
	zero_action = np.zeros((env.num_envs, action_len), dtype=np.float32)
	base_actions = []
	for _ in range(num_rollouts):
		env.reset()
		for _ in range(max_steps):
			base = env.get_base_action_flat()
			if base is not None:
				base_actions.append(base)
			_, reward, done, info = env.step(zero_action)
			if np.all(done):
				break
	return np.concatenate(base_actions, axis=0)


def collect_residual_basis_rollouts(model, env, num_steps, cfg):
	obs = env.reset()
	zero_action = np.zeros((cfg.env.n_envs, cfg.train.n_basis), dtype=np.float32)
	for _ in range(num_steps):
		scaled_action = model.policy.scale_action(zero_action)
		next_obs, reward, done, info = env.step(scaled_action)
		model.replay_buffer.add(
			obs=obs,
			next_obs=next_obs,
			action=scaled_action,
			reward=reward,
			done=done,
			infos=info,
		)
		obs = next_obs
	model.replay_buffer.final_offline_step()


def load_offline_data(model, offline_data_path, n_env):
	# this function should only be applied with dsrl_na
	offline_data = np.load(offline_data_path)
	obs = offline_data['states']
	next_obs = offline_data['states_next']
	actions = offline_data['actions']
	rewards = offline_data['rewards']
	terminals = offline_data['terminals']
	for i in range(int(obs.shape[0]/n_env)):
		model.replay_buffer.add(
					obs=obs[n_env*i:n_env*i+n_env],
					next_obs=next_obs[n_env*i:n_env*i+n_env],
					action=actions[n_env*i:n_env*i+n_env],
					reward=rewards[n_env*i:n_env*i+n_env],
					done=terminals[n_env*i:n_env*i+n_env],
					infos=[{}] * n_env,
				)
	model.replay_buffer.final_offline_step()