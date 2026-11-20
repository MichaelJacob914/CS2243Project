"""Furniture-Bench (IsaacGym, GPU-batched) as an SB3 VecEnv for train_adsrl.py.

``FurnitureVecEnv`` replaces DPPO's ``FurnitureRLSimEnvMultiStepWrapper`` (dsrl/dppo/env/gym_utils/
wrapper/furniture.py) + the robomimic ``ActionChunkWrapper``/``ObservationWrapperRobomimic`` stack,
with the same observation/action processing as DPPO and the same reward/episode semantics as the
robomimic pipeline in this repo:

- Observation: DPPO's ``process_obs`` -- robot_state with the quaternion converted to 6D rotation,
  concatenated with parts_poses, min-max normalized to [-1, 1] with normalization.pth and clamped to
  [-5, 5]. 58 dims for one_leg, 44 for lamp/round_table.
- Action: one chunk (act_steps x action_dim, flattened) in the base model's normalized [-1, 1] space;
  unnormalized with normalization.pth and executed as act_steps sim steps.
- Reward: like ObservationWrapperRobomimic + ActionChunkWrapper, the per-sim-step reward is
  ``success_t - reward_offset`` summed over the chunk, where success_t = "all parts of the task are
  assembled" (FurnitureRLSimEnv's sticky ``already_assembled`` flags, i.e. its ``done``). With
  reward_offset=1 a chunk is -8 before success and 0 once assembled.
  ``partial_reward=True`` (lamp, round_table: 2 pairs each) replaces success_t by the fraction of
  pairs assembled, ``already_assembled.mean()``, i.e. partial credit per part as in DPPO's
  "permanent" (dense) furniture reward, normalized so full assembly is worth the same as without it.
  For a 1-pair task (one_leg) the two rewards are identical.
- Success: ``info["chunk_success"]`` (every step) is True when all parts are assembled at the end of
  the chunk (flags are sticky, so: at any step so far in this episode). AdsrlCallback uses this flag,
  not the reward, for furniture, so partial credit is never counted as success.
- Episodes: fixed length, all envs in lockstep (``ceil(max_episode_steps / act_steps)`` chunks), as in
  DPPO. Success does not end an episode (as in robomimic here). At the last chunk every env reports
  done=True (a terminal, not a TimeLimit truncation -- same as ActionChunkWrapper), with
  ``terminal_observation``, ``episode`` and ``is_success`` in its info; then the whole batch is reset
  and the returned obs is the first obs of the next episode. Lockstep resets are needed because
  FurnitureRLSimEnv's lamp bulb logic assumes synchronized env_steps.

IsaacGym allows one sim per process, so train_adsrl.py evaluates on this same env (see
AdsrlCallback: evals run at episode boundaries over full episodes of the whole batch).

Data stays on the sim device until the SB3 boundary, where obs/reward (N x 58 floats) are copied to
numpy once per chunk; the action chunk is copied to the device once per chunk.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

# DPPO furniture task settings (dsrl/dppo/cfg/furniture/finetune/<task>/ft_ppo_diffusion_mlp.yaml)
OBS_CLAMP = 5.0


def load_furniture_normalizer(path: str, device) -> dict:
    """normalization.pth (a DPPO ``LinearNormalizer`` state dict) -> {key: (min, max)} on ``device``."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    stats: dict = {}
    for key, value in sd.items():
        assert key.startswith("stats."), key
        name, which = key[len("stats.") :].rsplit(".", 1)
        stats.setdefault(name, {})[which] = value.to(device=device, dtype=torch.float32)
    return {k: (v["min"], v["max"]) for k, v in stats.items()}


def normalize(x: torch.Tensor, stats) -> torch.Tensor:
    lo, hi = stats
    return 2 * (x - lo) / (hi - lo) - 1  # LinearNormalizer._normalize


def denormalize(x: torch.Tensor, stats) -> torch.Tensor:
    lo, hi = stats
    return (x + 1) / 2 * (hi - lo) + lo  # LinearNormalizer._denormalize


def make_furniture_sim(furniture: str, randomness: str, num_envs: int, max_env_steps: int, gpu_id: int = 0,
                       headless: bool = True, physx: Optional[dict] = None):  # fmt: skip
    """FurnitureRLSimEnv with the exact arguments of DPPO's ``make_async(env_type="furniture")``.
    Must be imported/created before anything else initializes CUDA through torch in this process.

    ``physx``: optional overrides of furniture_bench's ``sim_config["sim_params"].physx`` GPU buffer sizes
    (e.g. max_gpu_contact_pairs, default_buffer_size_multiplier, found_lost_pairs_capacity), applied
    before the sim is created. They size PhysX's GPU buffers and do not change the dynamics."""
    from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
    from furniture_bench.envs.observation import DEFAULT_STATE_OBS
    from furniture_bench.sim_config import sim_config

    for key, value in (physx or {}).items():
        assert hasattr(sim_config["sim_params"].physx, key), key
        setattr(sim_config["sim_params"].physx, key, type(getattr(sim_config["sim_params"].physx, key))(value))
        print(f"physx.{key} = {getattr(sim_config['sim_params'].physx, key)}", flush=True)

    return FurnitureRLSimEnv(
        act_rot_repr="rot_6d",
        action_type="pos",
        april_tags=False,
        concat_robot_state=True,
        ctrl_mode="diffik",
        obs_keys=DEFAULT_STATE_OBS,
        furniture=furniture,
        gpu_id=gpu_id,
        headless=headless,
        num_envs=num_envs,
        observation_space="state",
        randomness=randomness,
        max_env_steps=max_env_steps,
        record=False,
        pos_scalar=1,
        rot_scalar=1,
        stiffness=1_000,
        damping=200,
    )


def _default_robot_state_fn():
    from furniture_bench.controllers.control_utils import proprioceptive_quat_to_6d_rotation

    return proprioceptive_quat_to_6d_rotation


class FurnitureVecEnv(VecEnv):
    """SB3 VecEnv over a batched FurnitureRLSimEnv (see module docstring).

    ``sim`` needs: ``num_envs``, ``device``, ``reset() -> obs`` and ``step(a[N, A]) -> (obs, reward,
    done[N, 1], info)`` with obs = {"robot_state": [N, 14], "parts_poses": [N, P]} (the
    FurnitureRLSimEnv interface; tests pass a fake). ``robot_state_fn`` defaults to furniture_bench's
    ``proprioceptive_quat_to_6d_rotation``.
    """

    def __init__(
        self,
        sim,
        normalization_path: str,
        act_steps: int,
        action_dim: int,
        max_episode_steps: int,
        reward_offset: float = 1.0,
        partial_reward: bool = False,
        robot_state_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.sim = sim
        self.device = torch.device(sim.device) if not isinstance(sim.device, torch.device) else sim.device
        self.act_steps = int(act_steps)
        self.action_dim = int(action_dim)
        self.reward_offset = float(reward_offset)
        self.partial_reward = bool(partial_reward)
        self.chunks_per_episode = int(math.ceil(max_episode_steps / self.act_steps))
        self.robot_state_fn = robot_state_fn or _default_robot_state_fn()
        stats = load_furniture_normalizer(normalization_path, self.device)
        self.obs_stats, self.act_stats = stats["observations"], stats["actions"]
        assert self.act_stats[0].numel() == self.action_dim, (self.act_stats[0].shape, action_dim)
        obs_dim = int(self.obs_stats[0].numel())
        observation_space = spaces.Box(-OBS_CLAMP, OBS_CLAMP, (obs_dim,), dtype=np.float32)
        # the base model's normalized action space, tiled over the chunk (as ActionChunkWrapper)
        action_space = spaces.Box(-1.0, 1.0, (self.act_steps * self.action_dim,), dtype=np.float32)
        super().__init__(int(sim.num_envs), observation_space, action_space)
        self.obs_min = self.obs_stats[0].cpu().numpy()
        self.obs_max = self.obs_stats[1].cpu().numpy()
        self._actions: Optional[torch.Tensor] = None
        self.t = 0  # chunks into the current (synchronized) episode
        self.ep_return = torch.zeros(self.num_envs, device=self.device)
        self.ep_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ---- processing (DPPO FurnitureRLSimEnvMultiStepWrapper.process_obs, n_obs_steps=1) ----
    def process_obs(self, obs: dict) -> torch.Tensor:
        robot_state = self.robot_state_fn(obs["robot_state"])
        x = torch.cat([robot_state, obs["parts_poses"]], dim=-1).to(torch.float32)
        return torch.clamp(normalize(x, self.obs_stats), -OBS_CLAMP, OBS_CLAMP)

    def unnormalize_obs(self, nobs: np.ndarray) -> np.ndarray:
        lo, hi = self.obs_min, self.obs_max
        return (np.asarray(nobs) + 1) / 2 * (hi - lo) + lo

    # ---- VecEnv API ----
    def reset(self) -> np.ndarray:
        obs = self.sim.reset()
        self.t = 0
        self.ep_return.zero_()
        self.ep_success.zero_()
        self.reset_infos = [{} for _ in range(self.num_envs)]
        return self.process_obs(obs).cpu().numpy()

    def step_async(self, actions) -> None:
        a = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        self._actions = a.reshape(self.num_envs, self.act_steps, self.action_dim)

    @torch.no_grad()
    def step_wait(self):
        assert self._actions is not None, "step_async first"
        a = denormalize(self._actions, self.act_stats)
        self._actions = None
        reward = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.act_steps):
            obs, _, done, _ = self.sim.step(a[:, i])
            success = done.reshape(self.num_envs).bool()  # all parts assembled (sticky)
            if self.partial_reward:  # fraction of pairs assembled (sticky per pair)
                credit = self.sim.already_assembled.reshape(self.num_envs, -1).float().mean(1)
            else:
                credit = success.float()
            reward += credit - self.reward_offset
            self.ep_success |= success
        self.t += 1
        self.ep_return += reward
        obs_np = self.process_obs(obs).cpu().numpy()
        rew_np = reward.cpu().numpy()
        succ_now = success.cpu().numpy()
        if self.t < self.chunks_per_episode:
            infos = [{"chunk_success": bool(succ_now[i])} for i in range(self.num_envs)]
            return obs_np, rew_np, np.zeros(self.num_envs, dtype=bool), infos
        ret = self.ep_return.cpu().numpy()
        succ = self.ep_success.cpu().numpy()
        infos: List[dict] = [
            {"terminal_observation": obs_np[i], "episode": {"r": float(ret[i]), "l": self.t}, "is_success": bool(succ[i]),
             "chunk_success": bool(succ_now[i])}
            for i in range(self.num_envs)
        ]  # fmt: skip
        return self.reset(), rew_np, np.ones(self.num_envs, dtype=bool), infos

    def close(self) -> None:
        sim, self.sim = self.sim, None
        del sim

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        return [getattr(self, attr_name)] * len(self._indices(indices))

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> List[Any]:
        out = getattr(self, method_name)(*method_args, **method_kwargs)
        return [out] * len(self._indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return [False] * len(self._indices(indices))

    def _indices(self, indices) -> Sequence[int]:
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices


def make_furniture_vec_env(cfg) -> FurnitureVecEnv:
    """Builds the env from a train_adsrl config (cfg.env.furniture.*, cfg.normalization_path)."""
    fc = cfg.env.furniture
    sim = make_furniture_sim(
        furniture=fc.name, randomness=fc.randomness, num_envs=int(cfg.env.n_envs),
        max_env_steps=int(cfg.env.max_episode_steps), gpu_id=int(fc.get("gpu_id", 0)), headless=True,
        physx=dict(fc.get("physx") or {}),
    )  # fmt: skip
    return FurnitureVecEnv(
        sim, cfg.normalization_path, act_steps=cfg.act_steps, action_dim=cfg.action_dim,
        max_episode_steps=int(cfg.env.max_episode_steps), reward_offset=cfg.env.reward_offset,
        partial_reward=bool(fc.get("partial_reward", False)),
    )  # fmt: skip
