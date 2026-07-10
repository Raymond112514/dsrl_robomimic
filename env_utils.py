import os
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import sys
import gym
import gymnasium
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
import json

from dppo.env.gym_utils.wrapper import wrapper_dict
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def make_robomimic_env(render=False, env='square', normalization_path=None, low_dim_keys=None, dppo_path=None):
	wrappers = OmegaConf.create({
		'robomimic_lowdim': {
			'normalization_path': normalization_path,
			'low_dim_keys': low_dim_keys,
		},
	})
	obs_modality_dict = {
		"low_dim": (
			wrappers.robomimic_image.low_dim_keys
			if "robomimic_image" in wrappers
			else wrappers.robomimic_lowdim.low_dim_keys
		),
		"rgb": (
			wrappers.robomimic_image.image_keys
			if "robomimic_image" in wrappers
			else None
		),
	}
	if obs_modality_dict["rgb"] is None:
		obs_modality_dict.pop("rgb")
	ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
	robomimic_env_cfg_path = f'{dppo_path}/cfg/robomimic/env_meta/{env}.json'
	with open(robomimic_env_cfg_path, "r") as f:
		env_meta = json.load(f)
	env_meta["reward_shaping"] = False
	env = EnvUtils.create_env_from_metadata(
		env_meta=env_meta,
		render=False,
		render_offscreen=render,
		use_image_obs=False,
	)
	env.env.hard_reset = False
	for wrapper, args in wrappers.items():
		env = wrapper_dict[wrapper](env, **args)
	return env


class ObservationWrapperRobomimic(gym.Env):
	def __init__(
		self,
		env,
		reward_offset=1,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		self.reward_offset = reward_offset

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = raw_obs['state'].flatten()
		return obs

	def step(self, action):
		raw_obs, reward, done, info = self.env.step(action)
		reward = (reward - self.reward_offset)
		obs = raw_obs['state'].flatten()
		return obs, reward, done, info

	def render(self, **kwargs):
		return self.env.render()
	

class ObservationWrapperGym(gym.Env):
	def __init__(
		self,
		env,
		normalization_path,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		normalization = np.load(normalization_path)
		self.obs_min = normalization["obs_min"]
		self.obs_max = normalization["obs_max"]
		self.action_min = normalization["action_min"]
		self.action_max = normalization["action_max"]

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = self.normalize_obs(raw_obs)
		return obs

	def step(self, action):
		raw_action = self.unnormalize_action(action)
		raw_obs, reward, done, info = self.env.step(raw_action)
		obs = self.normalize_obs(raw_obs)
		return obs, reward, done, info

	def render(self, **kwargs):
		return self.env.render()
	
	def normalize_obs(self, obs):
		return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

	def unnormalize_action(self, action):
		action = (action + 1) / 2
		return action * (self.action_max - self.action_min) + self.action_min
	

class ActionChunkWrapper(gymnasium.Env):
	def __init__(self, env, cfg, max_episode_steps=300):
		self.max_episode_steps = max_episode_steps
		self.env = env
		self.act_steps = cfg.act_steps
		obs_dim = getattr(cfg, "base_obs_dim", cfg.obs_dim)
		self.action_space = spaces.Box(
			low=np.tile(env.action_space.low, cfg.act_steps),
			high=np.tile(env.action_space.high, cfg.act_steps),
			dtype=np.float32
		)
		self.observation_space = spaces.Box(
			low=-np.ones(obs_dim),
			high=np.ones(obs_dim),
			dtype=np.float32
		)
		self.count = 0

	def reset(self, seed=None):
		obs = self.env.reset(seed=seed)
		self.count = 0
		return obs, {}
	
	def step(self, action):
		if len(action.shape) == 1:
			action = action.reshape(self.act_steps, -1)
		obs_ = []
		reward_ = []
		done_ = []
		info_ = []
		done_i = False
		for i in range(action.shape[0]):
			self.count += 1
			obs_i, reward_i, done_i, info_i = self.env.step(action[i])
			obs_.append(obs_i)
			reward_.append(reward_i)
			done_.append(done_i)
			info_.append(info_i)
		obs = obs_[-1]
		reward = sum(reward_)
		done = np.max(done_)
		info = info_[-1]
		if self.count >= self.max_episode_steps:
			done = True
		if done:
			info['terminal_observation'] = obs
		return obs, reward, done, False, info

	def render(self):
		return self.env.render()
	
	def close(self):
		return
	

class DiffusionPolicyEnvWrapper(VecEnvWrapper):
	def __init__(self, env, cfg, base_policy):
		super().__init__(env)
		self.action_horizon = cfg.act_steps
		self.action_dim = cfg.action_dim
		self.action_space = spaces.Box(
			low=-cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
			high=cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
			dtype=np.float32
		)
		self.obs_dim = cfg.obs_dim
		self.observation_space = spaces.Box(
			low=-np.ones(self.obs_dim),
			high=np.ones(self.obs_dim),
			dtype=np.float32
		)
		self.env = env
		self.device = cfg.model.device
		self.base_policy = base_policy
		self.obs = None

	def step_async(self, actions):
		actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
		actions = actions.view(-1, self.action_horizon, self.action_dim)
		diffused_actions = self.base_policy(self.obs, actions)
		self.venv.step_async(diffused_actions)

	def step_wait(self):
		obs, rewards, dones, infos = self.venv.step_wait()
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy(), rewards, dones, infos

	def reset(self):
		obs = self.venv.reset()
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy()


class ResidualPolicyEnvWrapper(VecEnvWrapper):
	def __init__(self, env, cfg, base_policy):
		super().__init__(env)
		self.action_horizon = cfg.act_steps
		self.action_dim = cfg.action_dim
		self.base_obs_dim = cfg.base_obs_dim
		self.residual_scale = cfg.train.residual_scale
		self.base_noise_mode = cfg.train.get("base_noise_mode", "random")
		self.device = cfg.device
		self.base_policy = base_policy
		self.raw_obs = None
		self.last_base_action = None

		action_len = self.action_dim * self.action_horizon
		self.action_space = spaces.Box(
			low=-cfg.train.action_magnitude * np.ones(action_len),
			high=cfg.train.action_magnitude * np.ones(action_len),
			dtype=np.float32,
		)
		aug_obs_dim = self.base_obs_dim + action_len
		self.observation_space = spaces.Box(
			low=-np.inf * np.ones(aug_obs_dim),
			high=np.inf * np.ones(aug_obs_dim),
			dtype=np.float32,
		)

	def _sample_base_noise(self, batch_size):
		noise = torch.randn(
			batch_size, self.action_horizon, self.action_dim, device=self.device
		)
		if self.base_noise_mode == "zero":
			noise = torch.zeros_like(noise)
		return noise

	def _compute_base_action(self, raw_obs):
		noise = self._sample_base_noise(raw_obs.shape[0])
		return self.base_policy(raw_obs, noise, return_numpy=False)

	def _augment_obs(self, raw_obs, base_action):
		flat = base_action.reshape(base_action.shape[0], -1)
		if isinstance(raw_obs, torch.Tensor):
			return torch.cat([raw_obs, flat], dim=1)
		return np.concatenate([raw_obs, flat.detach().cpu().numpy()], axis=1)

	def _augment_terminal_obs(self, infos):
		for info in infos:
			if "terminal_observation" not in info:
				continue
			term_raw = torch.tensor(
				info["terminal_observation"][None],
				device=self.device,
				dtype=torch.float32,
			)
			term_base = self._compute_base_action(term_raw)
			info["terminal_observation"] = (
				self._augment_obs(term_raw, term_base)[0].detach().cpu().numpy()
			)

	def step_async(self, actions):
		residual = torch.tensor(actions, device=self.device, dtype=torch.float32)
		residual = residual.view(-1, self.action_horizon, self.action_dim)
		final_action = self.last_base_action + self.residual_scale * residual
		self.venv.step_async(final_action.detach().cpu().numpy())

	def step_wait(self):
		obs, rewards, dones, infos = self.venv.step_wait()
		self.raw_obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		self.last_base_action = self._compute_base_action(self.raw_obs)
		obs_out = self._augment_obs(self.raw_obs, self.last_base_action)
		self._augment_terminal_obs(infos)
		return obs_out.detach().cpu().numpy(), rewards, dones, infos

	def reset(self):
		obs = self.venv.reset()
		self.raw_obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		self.last_base_action = self._compute_base_action(self.raw_obs)
		obs_out = self._augment_obs(self.raw_obs, self.last_base_action)
		return obs_out.detach().cpu().numpy()

	def get_base_action_flat(self):
		if self.last_base_action is None:
			return None
		return (
			self.last_base_action.reshape(self.last_base_action.shape[0], -1)
			.detach()
			.cpu()
			.numpy()
		)


class ResidualBasisPolicyEnvWrapper(ResidualPolicyEnvWrapper):
	def __init__(self, env, cfg, base_policy):
		self.n_basis = cfg.train.n_basis
		self.basis_V = None
		super().__init__(env, cfg, base_policy)
		self.action_space = spaces.Box(
			low=-cfg.train.action_magnitude * np.ones(self.n_basis),
			high=cfg.train.action_magnitude * np.ones(self.n_basis),
			dtype=np.float32,
		)

	def set_basis(self, V):
		self.basis_V = np.asarray(V, dtype=np.float32)

	def _basis_to_residual(self, actions):
		actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
		if actions.ndim == 1:
			actions = actions.unsqueeze(0)
		action_len = self.action_horizon * self.action_dim
		if self.basis_V is None:
			return torch.zeros(
				actions.shape[0], action_len, device=self.device, dtype=torch.float32
			)
		V = torch.as_tensor(self.basis_V, device=self.device, dtype=torch.float32)
		return actions @ V.T

	def step_async(self, actions):
		residual_flat = self._basis_to_residual(actions)
		residual = residual_flat.view(-1, self.action_horizon, self.action_dim)
		final_action = self.last_base_action + self.residual_scale * residual
		self.venv.step_async(final_action.detach().cpu().numpy())

