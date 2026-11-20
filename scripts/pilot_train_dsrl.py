import os
import warnings
warnings.filterwarnings("ignore")
import math
import torch
import random
import wandb
import numpy as np
import json
import hydra
from omegaconf import OmegaConf
import gym
import sys
import time
# Pilot copy of dsrl/train_dsrl.py: d4rl import removed, total timesteps configurable,
# optional timing instrumentation. Run with cwd = dsrl/ (relative paths in the cfg).
DSRL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dsrl')
sys.path.insert(0, DSRL_DIR)
sys.path.append('./dppo')
 
from stable_baselines3 import SAC, DSRL
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env_utils import DiffusionPolicyEnvWrapper, ObservationWrapperRobomimic, ObservationWrapperGym, ActionChunkWrapper, make_robomimic_env
from utils import load_base_policy, load_offline_data, collect_rollouts, LoggingCallback

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = DSRL_DIR

	


TIMERS = {'init_eval': 0.0, 'init_rollout': 0.0, 'train': 0.0, 'n_train_calls': 0, 'n_grad_steps': 0, 'eval': 0.0, 'n_evals': 0}
T_START = [None]


def _sync():
	if torch.cuda.is_available():
		torch.cuda.synchronize()


def install_timers(model, logging_callback):
	"""Wrap model.train and callback.evaluate with wall-clock timers (no behavior change)."""
	orig_train, orig_eval = model.train, logging_callback.evaluate

	def timed_train(gradient_steps, batch_size=64):
		_sync(); t = time.perf_counter()
		out = orig_train(gradient_steps=gradient_steps, batch_size=batch_size)
		_sync(); TIMERS['train'] += time.perf_counter() - t
		TIMERS['n_train_calls'] += 1; TIMERS['n_grad_steps'] += gradient_steps
		if TIMERS['n_train_calls'] % 25 == 0:
			print('PILOT_PROGRESS ' + json.dumps(dict(t=time.time(), env_steps=model.num_timesteps * getattr(model, 'diffusion_act_chunk', 1),
				n_grad_steps=TIMERS['n_grad_steps'], train_s=TIMERS['train'])), flush=True)
		return out

	def timed_eval(agent, deterministic=False):
		t = time.perf_counter()
		out = orig_eval(agent, deterministic=deterministic)
		TIMERS['eval'] += time.perf_counter() - t; TIMERS['n_evals'] += 1
		return out

	model.train = timed_train
	logging_callback.evaluate = timed_eval


def install_eval_csv(logging_callback, path, cfg):
	"""Tee upstream's eval wandb.log calls into a CSV (upstream reports eval success only to wandb).
	env_steps = raw env steps incl. warm start (upstream's own counter counts the warm start in
	vec steps * n_envs, i.e. without the act_steps factor; we correct that so it matches train_adsrl)."""
	import csv
	orig_log = wandb.log
	use_wandb = cfg.use_wandb
	warm_fix = cfg.train.init_rollout_steps * cfg.env.n_envs * (cfg.act_steps - 1)
	t0 = time.time()
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

	def tee(d, *args, **kwargs):
		ev = {k: v for k, v in d.items() if k.startswith('eval/success_rate')}
		if ev:
			ts = logging_callback.total_timesteps
			env_steps = ts + (warm_fix if ts > 0 else 0)
			new = not os.path.exists(path)
			with open(path, 'a', newline='') as f:
				wr = csv.writer(f)
				if new:
					wr.writerow(['env_steps', 'upstream_timesteps', 'mode', 'success', 'wall_s'])
				for k, v in ev.items():
					mode = 'det' if k.endswith('deterministic') else 'stoch'
					wr.writerow([env_steps, ts, mode, float(v), round(time.time() - t0, 1)])
			print(f'[eval {mode}] env_steps={env_steps} success={float(list(ev.values())[0]):.3f}', flush=True)
		if use_wandb:
			orig_log(d, *args, **kwargs)

	wandb.log = tee
	logging_callback.use_wandb = True  # so evaluate() calls wandb.log (train logging is dropped if wandb is off)


def report_timers(model):
	# drop the instance-level wrapper so model.save() doesn't try to pickle the closure (and the SubprocVecEnv behind it)
	model.__dict__.pop('train', None)
	wall = time.perf_counter() - T_START[0]
	n_chunks = model.num_timesteps
	env_steps = n_chunks * getattr(model, 'diffusion_act_chunk', 1)
	learn_eval = TIMERS['eval'] - TIMERS['init_eval']
	rest = wall - TIMERS['train'] - learn_eval
	res = dict(wall_s=wall, sb3_timesteps=n_chunks, env_steps=env_steps,
		env_steps_per_s=env_steps / wall, env_steps_per_s_excl_eval=env_steps / (wall - learn_eval),
		ms_per_grad_step=1000 * TIMERS['train'] / max(TIMERS['n_grad_steps'], 1),
		frac_train=TIMERS['train'] / wall, frac_eval=learn_eval / wall, frac_env_other=rest / wall, **TIMERS)
	print('PILOT_TIMING ' + json.dumps(res), flush=True)


@hydra.main(
	config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None
)
def main(cfg: OmegaConf):
	OmegaConf.resolve(cfg)

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
			config=OmegaConf.to_container(cfg, resolve=True),
		)

	MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

	num_env = cfg.env.n_envs
	def make_env():
		if cfg.env_name in ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2']:
			env = gym.make(cfg.env_name)
			env = ObservationWrapperGym(env, cfg.normalization_path)
		elif cfg.env_name in ['lift', 'can', 'square', 'transport']:
			env = make_robomimic_env(env=cfg.env_name, normalization_path=cfg.normalization_path, low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path)
			env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
		env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
		return env

	base_policy = load_base_policy(cfg)
	env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
	if cfg.algorithm == 'dsrl_sac':
		env = DiffusionPolicyEnvWrapper(env, cfg, base_policy)
	env.seed(cfg.seed + 1)
	post_linear_modules = None
	if cfg.train.use_layer_norm:
		post_linear_modules = [torch.nn.LayerNorm]

	net_arch = []
	for _ in range(cfg.train.num_layers):
		net_arch.append(cfg.train.layer_size)
	policy_kwargs = dict(
		net_arch=dict(pi=net_arch, qf=net_arch),
		activation_fn=torch.nn.Tanh,
		log_std_init=0.0,
		post_linear_modules=post_linear_modules,
		n_critics=cfg.train.n_critics,
	)
	if cfg.algorithm == 'dsrl_sac':
		model = SAC(
			"MlpPolicy",
			env,
			learning_rate=cfg.train.actor_lr,
			buffer_size=20000000,      # Replay buffer size
			learning_starts=1,    # How many steps before learning starts (total steps for all env combined)
			batch_size=cfg.train.batch_size,
			tau=cfg.train.tau,                # Target network update rate
			gamma=cfg.train.discount,               # Discount factor
			train_freq=cfg.train.train_freq,             # Update the model every train_freq steps
			gradient_steps=cfg.train.utd,         # How many gradient steps to do at each update
			action_noise=None,        # No additional action noise
			optimize_memory_usage=False,
			ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,          # Automatic entropy tuning
			target_update_interval=1, # Update target network every interval
			target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,    # Automatic target entropy
			use_sde=False,
			sde_sample_freq=-1,
			tensorboard_log=cfg.logdir,
			verbose=1,
			policy_kwargs=policy_kwargs,
		)
	elif cfg.algorithm == 'dsrl_na':
		model = DSRL(
			"MlpPolicy",
			env,
			learning_rate=cfg.train.actor_lr,
			buffer_size=cfg.get('buffer_size', 10000000),      # Replay buffer size
			learning_starts=1,    # How many steps before learning starts (total steps for all env combined)
			batch_size=cfg.train.batch_size,
			tau=cfg.train.tau,                # Target network update rate
			gamma=cfg.train.discount,               # Discount factor
			train_freq=cfg.train.train_freq,             # Update the model every train_freq steps
			gradient_steps=cfg.train.utd,         # How many gradient steps to do at each update
			action_noise=None,        # No additional action noise
			optimize_memory_usage=False,
			ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,          # Automatic entropy tuning
			target_update_interval=1, # Update target network every interval
			target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,    # Automatic target entropy
			use_sde=False,
			sde_sample_freq=-1,
			tensorboard_log=cfg.logdir,
			verbose=1,
			policy_kwargs=policy_kwargs,
			diffusion_policy=base_policy,
			diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
			noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
			critic_backup_combine_type=cfg.train.critic_backup_combine_type,
		)

	checkpoint_callback = CheckpointCallback(
		save_freq=cfg.save_model_interval, 
		save_path=cfg.logdir+'/checkpoint/',
		name_prefix='ft_policy',
		save_replay_buffer=cfg.save_replay_buffer, 
		save_vecnormalize=True,
	)

	num_env_eval = cfg.env.n_eval_envs
	eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
	if cfg.algorithm == 'dsrl_sac':
		eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy)
	eval_env.seed(cfg.seed + num_env + 1) 

	logging_callback = LoggingCallback(
		action_chunk = cfg.act_steps, 
		eval_episodes = int(cfg.num_evals / num_env_eval), 
		log_freq=MAX_STEPS, 
		use_wandb=cfg.use_wandb, 
		eval_env=eval_env, 
		eval_freq=cfg.eval_interval,
		num_train_env=num_env,
		num_eval_env=num_env_eval,
		rew_offset=cfg.env.reward_offset,
		algorithm=cfg.algorithm,
		max_steps=MAX_STEPS,
		deterministic_eval=cfg.deterministic_eval,
	)

	eval_csv = cfg.get('eval_csv', None)
	if eval_csv:
		install_eval_csv(logging_callback, eval_csv, cfg)
	profile = cfg.get('profile', False)
	if profile:
		install_timers(model, logging_callback)
	logging_callback.evaluate(model, deterministic=False)
	if cfg.deterministic_eval:
		logging_callback.evaluate(model, deterministic=True)
	logging_callback.log_count += 1

	if cfg.load_offline_data:
		load_offline_data(model, cfg.offline_data_path, num_env)
	TIMERS['init_eval'] = TIMERS['eval']
	t_roll = time.perf_counter()
	if cfg.train.init_rollout_steps > 0:
		collect_rollouts(model, env, cfg.train.init_rollout_steps, base_policy, cfg)	
		logging_callback.set_timesteps(cfg.train.init_rollout_steps * num_env)

	callbacks = [checkpoint_callback, logging_callback]
	# Train the agent
	TIMERS['init_rollout'] = time.perf_counter() - t_roll
	if profile:
		print('PILOT_INIT ' + json.dumps(TIMERS), flush=True)
	T_START[0] = time.perf_counter()
	model.learn(
		total_timesteps=cfg.get('total_timesteps', 20000000),  # SB3 timesteps = action chunks summed over envs
		callback = callbacks
	)

	if profile:
		report_timers(model)

	# Save the final model
	if len(cfg.name) > 0:
		model.save(cfg.logdir+"/checkpoint/final")

	# Close environment and wandb
	env.close()
	if cfg.use_wandb:
		wandb.finish()


if __name__ == "__main__":
	main()
