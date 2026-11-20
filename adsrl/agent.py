"""DSRL-NA agents that steer the diffusion policy at denoising step k.

- ``DSRLFixedK``: upstream DSRL-NA with a constant steering step k. It overrides no learning
  logic: the diffusion policy handed to upstream ``DSRL`` is an adapter that calls
  ``SteeredBasePolicy(obs_aug, k, w)``, so every upstream diffusion call (``_sample_action``,
  ``predict_diffused``, the target backup and ``update_noise_critic``) uses k.
- ``DSRLAdaptiveK``: hybrid SAC over (k, w). A categorical head pi(k|s,x_T) and K squashed
  Gaussian heads pi(w|s,x_T,k) share one MLP trunk. The noise critic is Q^W(s,x_T,onehot(k),w).

steer.param=kick (set on the SteeredBasePolicy) makes w a perturbation of the implied noise, so
w = 0 decodes the base action at every k; the agents keep acting, learning and storing in w space and
only change the distillation distribution (``_SteerMixin._distill_w``). ``actor_zero_init`` zeroes the
actor's mean output layer, which together with kick makes the untrained deterministic actor the base policy.

Both agents also add checkpoint/resume helpers, a replay buffer whose sampling is O(batch)
and works once the buffer wraps, and optional TF32 / ``torch.compile`` flags (default off).
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import recursive_getattr
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.dsrl.dsrl import DSRL

from adsrl.trust_region import NoiseTrustRegion, sq_norm_per_dim

LOG_STD_MAX = 2
LOG_STD_MIN = -20
_SQUASH_EPS = 1e-6  # same epsilon as SB3's TanhBijector.log_prob_correction


# --------------------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------------------
class FastReplayBuffer(ReplayBuffer):
    """Uniform sampling with ``randint``.

    The fork's ``BaseBuffer.sample`` builds an O(N) probability vector on every call, and once
    the buffer wraps it raises (p has length ``pos`` but the support has length ``buffer_size``).
    Without offline mixing its distribution is uniform, which is what this samples.
    """

    def sample(self, batch_size: int, env=None):
        if self.offline_mix_ratio > 0 or self.optimize_memory_usage:
            return super().sample(batch_size, env=env)
        upper = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper, size=batch_size)
        return self._get_samples(batch_inds, env=env)


# --------------------------------------------------------------------------------------
# Feature extractor
# --------------------------------------------------------------------------------------
class DropXTExtractor(BaseFeaturesExtractor):
    """Keeps only the first ``keep_dim`` obs entries (the env state) and drops the appended x_T.
    For fixed k=0, where x_T never affects the action; it gives the networks upstream's inputs."""

    def __init__(self, observation_space: spaces.Box, keep_dim: int):
        super().__init__(observation_space, features_dim=int(keep_dim))
        self.keep_dim = int(keep_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations[..., : self.keep_dim]


XT_INPUTS = ("none", "actor_noise_critic", "all")


class SteerSACPolicy(SACPolicy):
    """SB3 SAC policy that controls which networks see the x_T part of the obs.

    ``xt_inputs``:
      - "all": actor, action critic Q(s,a) and noise critic see [s, x_T] (plain SACPolicy).
      - "actor_noise_critic": Q(s,a) and its target see only s. x_T is resampled independently
        every chunk and its effect is already in a, so Q(s,a) does not depend on it. The actor
        sees x_T. The agents build the noise critic with "all" (see ``noise_critic_policy_kwargs``).
      - "none": every network sees only s. Valid only for fixed k=0, where x_T never matters.
    ``state_dim`` is the length of s (x_T is appended after it).
    """

    def __init__(self, *args, xt_inputs: str = "all", state_dim: Optional[int] = None, **kwargs):
        assert xt_inputs in XT_INPUTS, xt_inputs
        assert xt_inputs == "all" or state_dim is not None, "state_dim is required to drop x_T"
        # set before nn.Module init: SACPolicy.__init__ calls _build -> make_critic
        object.__setattr__(self, "xt_inputs", xt_inputs)
        object.__setattr__(self, "state_dim", state_dim)
        if xt_inputs == "none":
            kwargs["features_extractor_class"] = DropXTExtractor
            kwargs["features_extractor_kwargs"] = dict(keep_dim=state_dim)
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor=None):
        if features_extractor is None and self.xt_inputs == "actor_noise_critic":
            features_extractor = DropXTExtractor(self.observation_space, self.state_dim)
        return super().make_critic(features_extractor=features_extractor)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(xt_inputs=self.xt_inputs, state_dim=self.state_dim)
        if self.xt_inputs == "none":  # re-derived in __init__
            data.pop("features_extractor_class", None)
            data.pop("features_extractor_kwargs", None)
        return data


def noise_critic_policy_kwargs(policy_kwargs: dict) -> dict:
    """policy_kwargs for building the noise critic Q^W: it sees x_T unless xt_inputs == "none"."""
    pk = dict(policy_kwargs)
    if pk.get("xt_inputs") == "actor_noise_critic":
        pk["xt_inputs"] = "all"
    return pk


# --------------------------------------------------------------------------------------
# Diffusion-policy adapters (upstream call signature -> SteeredBasePolicy)
# --------------------------------------------------------------------------------------
class FixedKDiffusion:
    """Upstream signature ``f(obs, noise[B,H,A], return_numpy=True)`` -> steered at constant k."""

    def __init__(self, steered_policy, k: int):
        self.steered_policy = steered_policy
        self.k = int(k)

    def __call__(self, obs, noise, return_numpy: bool = True):
        B = noise.shape[0]
        with th.no_grad():
            out = self.steered_policy(obs, self.k, noise.reshape(B, -1))
        if return_numpy:
            return out.detach().cpu().numpy()
        return out


class _NoUpstreamDiffusion:
    """Placed in ``DSRL.diffusion_policy`` for the adaptive agent so that any upstream code path
    that would call the sampler without k fails loudly instead of silently using k=0."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError("DSRLAdaptiveK must call steered_policy(obs, k, w) explicitly")


# --------------------------------------------------------------------------------------
# Hybrid actor
# --------------------------------------------------------------------------------------
class HybridActor(nn.Module):
    """pi(k|s) (categorical) and K squashed Gaussian heads pi(w|s,k) on a shared trunk.

    ``allowed_k`` restricts the categorical to a subset of the K steering steps (k_subset
    ablation). All per-k outputs are over the allowed set, in the order of ``allowed_k``;
    ``allowed_k[i]`` maps head index i to the actual step k.

    With ``freeze_k_uniform`` the categorical is fixed uniform over ``allowed_k`` and the
    logits head is not used.

    Squashed actions are in [-1, 1] (SB3's scaled action space); the caller unscales them to
    the env action-space bounds, exactly like upstream NA.
    """

    def __init__(
        self,
        obs_dim: int,
        noise_dim: int,
        num_k: int,
        net_arch: Sequence[int],
        activation_fn: type[nn.Module] = nn.Tanh,
        post_linear_modules: Optional[list[type[nn.Module]]] = None,
        allowed_k: Optional[Sequence[int]] = None,
        freeze_k_uniform: bool = False,
    ):
        super().__init__()
        allowed = list(range(num_k)) if allowed_k is None else sorted({int(k) for k in allowed_k})
        assert len(allowed) > 0 and all(0 <= k < num_k for k in allowed), (allowed, num_k)
        self.num_k = int(num_k)
        self.noise_dim = int(noise_dim)
        self.n_heads = len(allowed)
        self.freeze_k_uniform = bool(freeze_k_uniform)
        self.register_buffer("allowed_k", th.as_tensor(allowed, dtype=th.long))

        self.trunk = nn.Sequential(
            *create_mlp(obs_dim, -1, list(net_arch), activation_fn, post_linear_modules=post_linear_modules)
        )
        last = net_arch[-1] if len(net_arch) > 0 else obs_dim
        self.k_logits = nn.Linear(last, self.n_heads)
        self.mu = nn.Linear(last, self.n_heads * self.noise_dim)
        self.log_std = nn.Linear(last, self.n_heads * self.noise_dim)

    # ---- distribution parameters ----
    def forward(self, obs: th.Tensor):
        """Returns log pi(k) [B,K'], mean [B,K',D], log_std [B,K',D]."""
        h = self.trunk(obs)
        B = obs.shape[0]
        mean = self.mu(h).view(B, self.n_heads, self.noise_dim)
        log_std = th.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX).view(B, self.n_heads, self.noise_dim)
        if self.freeze_k_uniform:
            log_pk = th.full((B, self.n_heads), -float(np.log(self.n_heads)), device=obs.device, dtype=mean.dtype)
        else:
            log_pk = F.log_softmax(self.k_logits(h), dim=-1)
        return log_pk, mean, log_std

    @staticmethod
    def _squashed_sample(mean, log_std, deterministic: bool = False):
        """Reparameterized tanh-Gaussian sample and its log-prob (summed over the last dim)."""
        if deterministic:
            u = mean
            eps = th.zeros_like(mean)
        else:
            eps = th.randn_like(mean)
            u = mean + log_std.exp() * eps
        w = th.tanh(u)
        log_prob = (-0.5 * eps.pow(2) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        log_prob = log_prob - th.log(1 - w.pow(2) + _SQUASH_EPS).sum(-1)
        return w, log_prob

    def all_heads(self, obs: th.Tensor) -> dict[str, th.Tensor]:
        """Reparameterized w_k from every head (for the exact sum over k in the actor loss).

        probs/log_pk: [B,K'], w: [B,K',D] (squashed, in [-1,1]), log_prob_w: [B,K'].
        """
        log_pk, mean, log_std = self(obs)
        w, log_prob_w = self._squashed_sample(mean, log_std)
        return {"log_pk": log_pk, "probs": log_pk.exp(), "w": w, "log_prob_w": log_prob_w}

    def sample(self, obs: th.Tensor, deterministic: bool = False) -> dict[str, th.Tensor]:
        """One (k, w) per row.

        Deterministic: argmax k and the tanh of the mean w. Under ``freeze_k_uniform`` argmax is
        meaningless, so k is still drawn uniformly and only w is deterministic.

        Returns k (actual step, LongTensor [B]), idx (head index [B]), w [B,D] squashed,
        log_prob_w [B], probs/log_pk [B,K'].
        """
        log_pk, mean, log_std = self(obs)
        probs = log_pk.exp()
        if deterministic and not self.freeze_k_uniform:
            idx = probs.argmax(dim=-1)
        else:
            idx = th.multinomial(probs, 1).squeeze(-1)
        gather = idx.view(-1, 1, 1).expand(-1, 1, self.noise_dim)
        mean_k = mean.gather(1, gather).squeeze(1)
        log_std_k = log_std.gather(1, gather).squeeze(1)
        w, log_prob_w = self._squashed_sample(mean_k, log_std_k, deterministic=deterministic)
        return {
            "k": self.allowed_k[idx],
            "idx": idx,
            "w": w,
            "log_prob_w": log_prob_w,
            "probs": probs,
            "log_pk": log_pk,
        }

    def onehot(self, k: th.Tensor) -> th.Tensor:
        """One-hot over all K steps (not just the allowed ones), so the critic layout is fixed."""
        return F.one_hot(k, self.num_k).to(th.float32)


# --------------------------------------------------------------------------------------
# Shared functionality
# --------------------------------------------------------------------------------------
class _SteerMixin:
    """Checkpointing, efficiency flags and helpers shared by both agents."""

    steered_policy: Any
    use_tf32: bool
    compile_networks: bool
    compile_mode: str

    def _init_steer_common(self, steered_policy, use_tf32: bool, compile_networks: bool, compile_mode: str,
                           actor_zero_init: bool = False, actor_log_std_init: Optional[float] = None):
        self.steered_policy = steered_policy
        self.actor_zero_init = bool(actor_zero_init)
        self.actor_log_std_init = None if actor_log_std_init is None else float(actor_log_std_init)
        self.use_tf32 = bool(use_tf32)
        self.compile_networks = bool(compile_networks)
        self.compile_mode = compile_mode
        if self.use_tf32:
            th.backends.cuda.matmul.allow_tf32 = True
            th.backends.cudnn.allow_tf32 = True

    @property
    def kick(self) -> bool:
        """steer.param == "kick" (read from the steered policy; test stubs without ``param`` are absolute)."""
        return getattr(self.steered_policy, "param", "absolute") == "kick"

    def _distill_w(self, B: int) -> th.Tensor:
        """Noise for distilling Q^W, [B, D] in env noise units. Absolute: w ~ N(0, I), the base policy's
        noise (upstream). Kick: Delta ~ Uniform[-1, 1]^D. Delta = 0 is the base action, so N(0, I) has no
        special meaning in kick space; the uniform box is the warm-up exploration distribution and covers
        every Delta the tanh-squashed actor can output (a tanh(N(mu, sigma)) actor concentrates inside it)."""
        D = self.diffusion_act_chunk * self.diffusion_act_dim
        if self.kick:
            return th.rand(B, D, device=self.device) * 2 - 1
        return th.randn(B, D, device=self.device)

    def _zero_init_actor_mean(self) -> None:
        """steer.actor_zero_init: zero weight and bias of the actor's mean output layer (all K' Gaussian heads
        of the hybrid actor; its k-logits and log-std heads keep their init). Deterministic actor -> w = 0."""
        if not self.actor_zero_init:
            return
        head = self.hybrid_actor.mu if hasattr(self, "hybrid_actor") else self.actor.mu
        with th.no_grad():
            head.weight.zero_()
            head.bias.zero_()

    def _init_actor_log_std(self) -> None:
        """steer.actor_log_std_init: log_std head weight = 0, bias = the value (all K' heads of the hybrid actor),
        so the initial policy std is exp(value) everywhere. SB3's log_std_init only applies to gSDE."""
        if self.actor_log_std_init is None:
            return
        head = self.hybrid_actor.log_std if hasattr(self, "hybrid_actor") else self.actor.log_std
        with th.no_grad():
            head.weight.zero_()
            head.bias.fill_(self.actor_log_std_init)

    def _modules_to_compile(self) -> list[nn.Module]:
        return [self.critic, self.critic_target, self.critic_noise, self.actor]

    # ---- noise trust region (adsrl/trust_region.py; off unless a config dict is given) ----
    def _setup_trust_region(self) -> None:
        tr = getattr(self, "_trust_region_cfg", None)
        self.trust_region = None
        if tr:
            self.trust_region = NoiseTrustRegion(tr["bound"], tr["lr"], tr.get("init", 0.0), device=self.device)

    def _trust_region_tensors(self) -> list[str]:
        return ["trust_region.lam"] if getattr(self, "trust_region", None) is not None else []

    def _maybe_compile(self) -> None:
        """In-place ``nn.Module.compile`` keeps parameter identity and state_dict keys, so optimizers,
        polyak updates and checkpoints are unaffected."""
        if not self.compile_networks:
            return
        for m in self._modules_to_compile():
            if hasattr(m, "compile"):
                m.compile(mode=self.compile_mode)
            else:  # torch < 2.2
                warnings.warn("nn.Module.compile unavailable; skipping torch.compile")
                return

    # Tensor versions of policy.scale_action / unscale_action (avoid host round-trips).
    def _bounds(self):
        if not hasattr(self, "_act_low_t"):
            self._act_low_t = th.as_tensor(self.action_space.low, device=self.device, dtype=th.float32)
            self._act_high_t = th.as_tensor(self.action_space.high, device=self.device, dtype=th.float32)
        return self._act_low_t, self._act_high_t

    def _unscale_t(self, scaled: th.Tensor) -> th.Tensor:
        low, high = self._bounds()
        return low + 0.5 * (scaled + 1.0) * (high - low)

    def _scale_t(self, action: th.Tensor) -> th.Tensor:
        low, high = self._bounds()
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def _combine_q(self, q_values: th.Tensor) -> th.Tensor:
        """q_values: [..., n_critics] -> [..., 1]."""
        if self.critic_backup_combine_type == "min":
            return q_values.min(dim=-1, keepdim=True)[0]
        if self.critic_backup_combine_type == "mean":
            return q_values.mean(dim=-1, keepdim=True)
        raise ValueError(self.critic_backup_combine_type)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["diffusion_policy", "steered_policy"]  # noqa: RUF005

    # ---- checkpoint / resume ----
    def save_checkpoint(self, path: str, extra: Optional[dict] = None, save_replay_buffer: bool = True) -> None:
        """Saves network/optimizer state, temperatures, counters and RNG states to ``path`` (.pt)
        and the replay buffer to ``<path stem>_replay_buffer.pkl``. Writes are atomic (tmp + rename).
        ``model.save`` is not used because it would pickle the diffusion model, and SB3 ``load``
        cannot rebuild these classes without the diffusion policy."""
        state_dicts, tensors = self._get_torch_save_params()
        payload = {
            "state_dicts": {n: recursive_getattr(self, n).state_dict() for n in state_dicts},
            "tensors": {n: recursive_getattr(self, n).detach().cpu() for n in tensors},
            "num_timesteps": self.num_timesteps,
            "n_updates": self._n_updates,
            "episode_num": self._episode_num,
            "rng": {
                "numpy": np.random.get_state(),
                "torch": th.get_rng_state(),
                "cuda": th.cuda.get_rng_state_all() if th.cuda.is_available() else None,
            },
            "extra": extra or {},
        }
        tmp = path + ".tmp"
        th.save(payload, tmp)
        os.replace(tmp, path)
        if save_replay_buffer and self.replay_buffer is not None:
            rb_path = self.replay_buffer_path(path)
            rb_tmp = rb_path[: -len(".pkl")] + "_tmp.pkl"
            self.save_replay_buffer(rb_tmp)
            os.replace(rb_tmp, rb_path)

    @staticmethod
    def replay_buffer_path(ckpt_path: str) -> str:
        return os.path.splitext(ckpt_path)[0] + "_replay_buffer.pkl"

    def load_checkpoint(self, path: str, load_replay_buffer: bool = True) -> dict:
        payload = th.load(path, map_location=self.device, weights_only=False)
        for name, sd in payload["state_dicts"].items():
            recursive_getattr(self, name).load_state_dict(sd)
        with th.no_grad():
            for name, value in payload["tensors"].items():
                # in place, so optimizers keep pointing at the same tensor
                recursive_getattr(self, name).copy_(value.to(self.device))
        self.num_timesteps = payload["num_timesteps"]
        self._n_updates = payload["n_updates"]
        self._episode_num = payload["episode_num"]
        rng = payload.get("rng", {})
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("torch") is not None:
            th.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") is not None and th.cuda.is_available():
            try:
                th.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
            except RuntimeError:
                pass  # different GPU count
        if load_replay_buffer:
            rb_path = self.replay_buffer_path(path)
            if os.path.exists(rb_path):
                self.load_replay_buffer(rb_path)
            else:
                warnings.warn(f"no replay buffer at {rb_path}; resuming with an empty buffer")
        return payload["extra"]


def _policy_net_arch(policy_kwargs: dict) -> list[int]:
    arch = policy_kwargs.get("net_arch", [256, 256])
    if isinstance(arch, dict):
        return list(arch["pi"])
    return list(arch)


# --------------------------------------------------------------------------------------
# Fixed k
# --------------------------------------------------------------------------------------
class DSRLFixedK(_SteerMixin, DSRL):
    """Upstream DSRL-NA where every diffusion call steers at a constant step k.

    Overrides: ``__init__`` (wraps the steered policy), ``_setup_model`` (efficiency flags),
    ``_get_torch_save_params`` (adds the noise-critic optimizer), ``_excluded_save_params``,
    ``train`` (calls upstream, then snapshots the train/* stats) and ``predict_diffused``
    (upstream logic, also records w in ``last_w`` for eval dumps). The learning code is upstream's.
    """

    def __init__(
        self,
        policy,
        env,
        *,
        steered_policy,
        k: int = 0,
        horizon: int,
        act_dim: int,
        use_tf32: bool = False,
        compile_networks: bool = False,
        compile_mode: str = "default",
        replay_buffer_class=None,
        trust_region: Optional[dict] = None,
        actor_zero_init: bool = False,
        actor_log_std_init: Optional[float] = None,
        **kwargs,
    ):
        self._init_steer_common(steered_policy, use_tf32, compile_networks, compile_mode, actor_zero_init,
                                actor_log_std_init)
        self._trust_region_cfg = trust_region
        self.k = int(k)
        xt = (kwargs.get("policy_kwargs") or {}).get("xt_inputs", "all")
        assert xt != "none" or self.k == 0, "xt_inputs='none' is only valid at fixed k=0"
        assert xt != "none" or not self.kick, "xt_inputs='none' is invalid with kick (the action depends on x_T)"
        self.last_k = self.k  # for eval logging
        self.last_w: Optional[np.ndarray] = None
        self.last_probs = None
        super().__init__(
            policy,
            env,
            diffusion_policy=FixedKDiffusion(steered_policy, self.k),
            diffusion_act_dim=(horizon, act_dim),
            replay_buffer_class=replay_buffer_class or FastReplayBuffer,
            **kwargs,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.policy_kwargs.get("xt_inputs") == "actor_noise_critic":
            # upstream built critic_noise from policy_kwargs (x_T dropped); Q^W must see x_T
            self.critic_noise = self.policy_class(
                self.observation_space, self.action_space, self.lr_schedule,
                **noise_critic_policy_kwargs(self.policy_kwargs),
            ).critic.to(self.device)  # fmt: skip
        self._setup_trust_region()
        self._zero_init_actor_mean()
        self._init_actor_log_std()
        self._maybe_compile()

    def _get_torch_save_params(self):
        state_dicts, tensors = super()._get_torch_save_params()
        return state_dicts + ["critic_noise.optimizer"], tensors + self._trust_region_tensors()

    def pop_k_counts(self) -> Optional[np.ndarray]:
        return None

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.trust_region is None:
            super().train(gradient_steps, batch_size)  # upstream DSRL-NA, unchanged
        else:
            self._train_trust_region(gradient_steps, batch_size)
        # snapshot: SB3 clears logger.name_to_value on every dump
        self.last_train_stats = {k: v for k, v in self.logger.name_to_value.items() if k.startswith("train/")}

    def _train_trust_region(self, gradient_steps: int, batch_size: int) -> None:
        """Upstream ``DSRL.train`` plus the noise trust-region term in the actor loss and the
        lambda update. Only used when ``trust_region`` is set, so the default path stays upstream's.
        Differences from upstream besides the penalty: the next-noise unscale is done on-device
        (same values as upstream's numpy round trip); no gSDE (unused here)."""
        self.policy.set_training_mode(True)
        self.critic_noise.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.critic_noise.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        logs: dict[str, list] = {k: [] for k in ["ent_coef", "ent_coef_loss", "actor_loss", "critic_loss",
                                                 "noise_critic_loss", "tr_lambda", "tr_w_sq"]}  # fmt: skip
        if self.actor_gradient_steps < 0:
            actor_gradient_idx = np.arange(gradient_steps)
        else:
            actor_gradient_idx = np.linspace(int(gradient_steps / self.actor_gradient_steps) - 1, gradient_steps - 1,
                                             self.actor_gradient_steps, dtype=int)  # fmt: skip
        chunk = (-1, self.diffusion_act_chunk, self.diffusion_act_dim)
        for gradient_step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            actions_pi, log_prob = self.actor.action_log_prob(data.observations)
            log_prob = log_prob.reshape(-1, 1)
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                logs["ent_coef_loss"].append(ent_coef_loss.item())
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            else:
                ent_coef = self.ent_coef_tensor
            logs["ent_coef"].append(ent_coef.item())

            with th.no_grad():
                next_w, next_log_prob = self.actor.action_log_prob(data.next_observations)
                next_a = self.diffusion_policy(data.next_observations, self._unscale_t(next_w).reshape(chunk),
                                               return_numpy=False).reshape(next_w.shape[0], -1)  # fmt: skip
                next_q = self._combine_q(th.cat(self.critic_target(data.next_observations, next_a), dim=1))
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = data.rewards + (1 - data.dones) * self.gamma * next_q
            current_q = self.critic(data.observations, data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            logs["critic_loss"].append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if gradient_step in actor_gradient_idx:
                q_pi = self._combine_q(th.cat(self.critic_noise(data.observations, actions_pi), dim=1))
                w_sq = sq_norm_per_dim(self._unscale_t(actions_pi))
                actor_loss = (ent_coef * log_prob - q_pi).mean() + self.trust_region.penalty(w_sq)
                logs["actor_loss"].append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                self.trust_region.update(w_sq.mean().item())
                logs["tr_lambda"].append(self.trust_region.lam.item())
                logs["tr_w_sq"].append(w_sq.mean().item())

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        for _ in range(self.noise_critic_grad_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            loss = self.update_noise_critic(data)
            logs["noise_critic_loss"].append(loss.item())
            self.critic_noise.optimizer.zero_grad()
            loss.backward()
            self.critic_noise.optimizer.step()
        self.critic_noise.set_training_mode(False)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, vals in logs.items():
            if vals:
                self.logger.record(f"train/{key}", float(np.mean(vals)))

    def update_noise_critic(self, replay_data):
        """Upstream (w ~ N(0, I)) in absolute mode. Kick: Delta ~ ``_distill_w``, same loss as upstream."""
        if not self.kick:
            return super().update_noise_critic(replay_data)
        obs = replay_data.observations
        B = obs.shape[0]
        with th.no_grad():
            w = self._distill_w(B)
            a = self.diffusion_policy(obs, w.reshape(B, self.diffusion_act_chunk, self.diffusion_act_dim),
                                      return_numpy=False).reshape(B, -1)  # fmt: skip
            target = self.critic(obs, a)
        pred = self.critic_noise(obs, self._scale_t(w))
        return sum(0.5 * F.mse_loss(t.detach(), p) for t, p in zip(target, pred))

    def predict_diffused(self, observation, state=None, episode_start=None, deterministic: bool = False):
        """Same as upstream, but also records the noise w in ``self.last_w`` for eval dumps."""
        unscaled, second = self.policy.predict(observation, state, episode_start, deterministic)
        self.last_w = np.asarray(unscaled, dtype=np.float32).reshape(len(unscaled), -1)
        obs = th.as_tensor(observation, device=self.device, dtype=th.float32)
        w = th.as_tensor(self.last_w, device=self.device)
        action = self.diffusion_policy(obs, w.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim), return_numpy=False)
        return action.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim).cpu().numpy(), second


# --------------------------------------------------------------------------------------
# Adaptive k (hybrid SAC)
# --------------------------------------------------------------------------------------
class DSRLAdaptiveK(_SteerMixin, DSRL):
    """Hybrid SAC over (k, w).

    Actor loss (exact sum over the allowed k):
        sum_k pi(k)[a_w log pi(w_k|k) - Q^W(s,k,w_k)] + a_k sum_k pi(k) log pi(k)
    Temperatures: a_w (upstream ``ent_coef``/``target_entropy``) and a_k with target entropy
    rho*log K' (K' = number of allowed k).
    k_entropy_bias='detach': pi(k) in the a_w log pi(w_k|k) term is stop-gradiented, so pi(k) is trained only
    by Q and a_k H(pi_k) (no bonus for k whose w head is high-entropy); the w heads' gradients are unchanged.
    'keep' (default) is the loss above.
    Target backup: (k', w') ~ pi; V(s') = minQ(s', a') - a_w log pi(w'|k') + a_k H(pi(.|s')).
    Distillation: k ~ Uniform(allowed), w ~ N(0, I) (kick: Delta ~ Uniform[-1, 1]^D, see ``_distill_w``).

    Overrides: ``__init__``, ``_setup_model``, ``train``, ``update_noise_critic``,
    ``_sample_action``, ``predict_diffused``, ``_get_torch_save_params``,
    ``_excluded_save_params``, ``_create_aliases``. The upstream Gaussian actor inside
    ``self.policy`` is built but never used; ``self.actor`` is the HybridActor.
    """

    def __init__(
        self,
        policy,
        env,
        *,
        steered_policy,
        num_k: int,
        horizon: int,
        act_dim: int,
        rho: float = 0.5,
        k_ent_coef: Union[str, float] = "auto",
        freeze_k_uniform: bool = False,
        k_subset: Optional[Sequence[int]] = None,
        k_entropy_bias: str = "keep",
        use_tf32: bool = False,
        compile_networks: bool = False,
        compile_mode: str = "default",
        replay_buffer_class=None,
        trust_region: Optional[dict] = None,
        actor_zero_init: bool = False,
        actor_log_std_init: Optional[float] = None,
        **kwargs,
    ):
        self._init_steer_common(steered_policy, use_tf32, compile_networks, compile_mode, actor_zero_init,
                                actor_log_std_init)
        self._trust_region_cfg = trust_region
        assert (kwargs.get("policy_kwargs") or {}).get("xt_inputs", "all") != "none", "adaptive k needs x_T"
        assert k_entropy_bias in ("keep", "detach"), k_entropy_bias
        self.k_entropy_bias = k_entropy_bias
        self.num_k = int(num_k)
        self.rho = float(rho)
        self.k_ent_coef = k_ent_coef
        self.freeze_k_uniform = bool(freeze_k_uniform)
        self.k_subset = None if k_subset is None else [int(k) for k in k_subset]
        self.last_k: Optional[np.ndarray] = None
        self.last_w: Optional[np.ndarray] = None
        self.last_probs: Optional[np.ndarray] = None
        self._k_counts = np.zeros(self.num_k, dtype=np.int64)
        self.last_mean_k_probs: Optional[np.ndarray] = None
        super().__init__(
            policy,
            env,
            diffusion_policy=_NoUpstreamDiffusion(),
            diffusion_act_dim=(horizon, act_dim),
            replay_buffer_class=replay_buffer_class or FastReplayBuffer,
            **kwargs,
        )

    # ---- model ----
    def _create_aliases(self) -> None:
        super()._create_aliases()
        if hasattr(self, "hybrid_actor"):
            self.actor = self.hybrid_actor

    def _setup_model(self) -> None:
        super()._setup_model()  # policy (critic, critic_target), a_w, upstream critic_noise
        noise_dim = int(np.prod(self.action_space.shape))
        obs_dim = int(np.prod(self.observation_space.shape))
        pk = self.policy_kwargs

        self.hybrid_actor = HybridActor(
            obs_dim,
            noise_dim,
            self.num_k,
            _policy_net_arch(pk),
            activation_fn=pk.get("activation_fn", nn.ReLU),
            post_linear_modules=pk.get("post_linear_modules"),
            allowed_k=self.k_subset,
            freeze_k_uniform=self.freeze_k_uniform,
        ).to(self.device)
        self.hybrid_actor.optimizer = th.optim.Adam(self.hybrid_actor.parameters(), lr=self.lr_schedule(1))
        self.actor = self.hybrid_actor

        # Q^W(s, x_T, onehot(k), w): a ContinuousCritic whose "action" is [onehot(k), w].
        kw_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_k + noise_dim,), dtype=np.float32)
        self.critic_noise = self.policy_class(
            self.observation_space, kw_space, self.lr_schedule, **noise_critic_policy_kwargs(pk)
        ).critic.to(self.device)

        # a_k
        self.n_allowed_k = self.hybrid_actor.n_heads
        self.target_entropy_k = self.rho * float(np.log(self.n_allowed_k))
        self.log_k_ent_coef = None
        self.k_ent_coef_optimizer = None
        self.k_ent_coef_tensor = th.zeros(1, device=self.device)
        if self.freeze_k_uniform:
            pass  # entropy of a frozen uniform pi(k) is constant; no a_k
        elif isinstance(self.k_ent_coef, str) and self.k_ent_coef.startswith("auto"):
            init = float(self.k_ent_coef.split("_")[1]) if "_" in self.k_ent_coef else 1.0
            self.log_k_ent_coef = th.log(th.ones(1, device=self.device) * init).requires_grad_(True)
            self.k_ent_coef_optimizer = th.optim.Adam([self.log_k_ent_coef], lr=self.lr_schedule(1))
        else:
            self.k_ent_coef_tensor = th.tensor(float(self.k_ent_coef), device=self.device)
        self._setup_trust_region()
        self._zero_init_actor_mean()
        self._init_actor_log_std()
        self._maybe_compile()

    def _alpha_w(self) -> th.Tensor:
        if self.log_ent_coef is not None:
            return th.exp(self.log_ent_coef.detach())
        return self.ent_coef_tensor

    def _alpha_k(self) -> th.Tensor:
        if self.log_k_ent_coef is not None:
            return th.exp(self.log_k_ent_coef.detach())
        return self.k_ent_coef_tensor

    def _q_noise(self, obs: th.Tensor, k: th.Tensor, w_scaled: th.Tensor) -> th.Tensor:
        """[B, n_critics]"""
        return th.cat(self.critic_noise(obs, th.cat([self.hybrid_actor.onehot(k), w_scaled], dim=-1)), dim=1)

    def _decode(self, obs: th.Tensor, k: th.Tensor, w_scaled: th.Tensor) -> th.Tensor:
        """Diffused action chunk [B, H*A] for squashed noise w (unscaled to the action bounds)."""
        a = self.steered_policy(obs, k, self._unscale_t(w_scaled))
        return a.reshape(obs.shape[0], -1).to(self.device, th.float32)

    # ---- learning ----
    def actor_loss(self, obs: th.Tensor, alpha_w: th.Tensor, alpha_k: th.Tensor):
        """Exact-sum hybrid actor loss. Returns (loss, aux) where aux holds detached stats."""
        out = self.hybrid_actor.all_heads(obs)
        B, Kp, D = out["w"].shape
        obs_rep = obs.unsqueeze(1).expand(B, Kp, obs.shape[-1]).reshape(B * Kp, -1)
        k_rep = self.hybrid_actor.allowed_k.unsqueeze(0).expand(B, Kp).reshape(-1)
        q = self._combine_q(self._q_noise(obs_rep, k_rep, out["w"].reshape(B * Kp, D))).view(B, Kp)
        probs, log_pk, log_pw = out["probs"], out["log_pk"], out["log_prob_w"]
        probs_w = probs.detach() if self.k_entropy_bias == "detach" else probs
        loss = (probs_w * alpha_w * log_pw - probs * q).sum(-1).mean()
        if not self.freeze_k_uniform:
            loss = loss + alpha_k * (probs * log_pk).sum(-1).mean()
        # E_k ||w_k||^2/D per row (w in env noise units), for the optional trust region
        w_sq = (probs * sq_norm_per_dim(self._unscale_t(out["w"]))).sum(-1)
        if self.trust_region is not None:
            loss = loss + self.trust_region.penalty(w_sq)
        aux = {
            "w_sq": w_sq.detach(),
            "w_log_prob": (probs * log_pw).sum(-1).detach(),  # E_k log pi(w_k|k), per row
            "k_entropy": -(probs * log_pk).sum(-1).detach(),
            "probs": probs.detach(),
        }
        return loss, aux

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self.critic_noise.set_training_mode(True)
        self.hybrid_actor.train(True)
        optimizers = [self.hybrid_actor.optimizer, self.critic.optimizer, self.critic_noise.optimizer]
        for opt in (self.ent_coef_optimizer, self.k_ent_coef_optimizer):
            if opt is not None:
                optimizers.append(opt)
        self._update_learning_rate(optimizers)

        logs: dict[str, list] = {
            k: []
            for k in [
                "ent_coef",
                "k_ent_coef",
                "actor_loss",
                "critic_loss",
                "noise_critic_loss",
                "ent_coef_loss",
                "k_ent_coef_loss",
                "k_entropy",
                "tr_lambda",
                "tr_w_sq",
            ]
        }
        probs_acc = []

        if self.actor_gradient_steps < 0:
            actor_gradient_idx = np.arange(gradient_steps)
        else:
            actor_gradient_idx = np.linspace(
                int(gradient_steps / self.actor_gradient_steps) - 1,
                gradient_steps - 1,
                self.actor_gradient_steps,
                dtype=int,
            )

        for gradient_step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            alpha_w, alpha_k = self._alpha_w(), self._alpha_k()
            logs["ent_coef"].append(alpha_w.item())
            logs["k_ent_coef"].append(alpha_k.item())

            # ---- critic Q(s, a) ----
            with th.no_grad():
                nxt = self.hybrid_actor.sample(data.next_observations)
                next_actions = self._decode(data.next_observations, nxt["k"], nxt["w"])
                next_q = self._combine_q(th.cat(self.critic_target(data.next_observations, next_actions), dim=1))
                next_q = next_q - alpha_w * nxt["log_prob_w"].reshape(-1, 1)
                if not self.freeze_k_uniform:
                    next_h_k = -(nxt["probs"] * nxt["log_pk"]).sum(-1, keepdim=True)
                    next_q = next_q + alpha_k * next_h_k
                target_q = data.rewards + (1 - data.dones) * self.gamma * next_q

            current_q = self.critic(data.observations, data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            logs["critic_loss"].append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # ---- actor + temperatures ----
            if gradient_step in actor_gradient_idx:
                actor_loss, aux = self.actor_loss(data.observations, alpha_w, alpha_k)
                logs["actor_loss"].append(actor_loss.item())
                self.hybrid_actor.optimizer.zero_grad()
                actor_loss.backward()
                self.hybrid_actor.optimizer.step()
                logs["k_entropy"].append(aux["k_entropy"].mean().item())
                probs_acc.append(aux["probs"].mean(0))
                if self.trust_region is not None:
                    self.trust_region.update(aux["w_sq"].mean().item())
                    logs["tr_lambda"].append(self.trust_region.lam.item())
                    logs["tr_w_sq"].append(aux["w_sq"].mean().item())

                if self.ent_coef_optimizer is not None:
                    loss_aw = -(self.log_ent_coef * (aux["w_log_prob"] + self.target_entropy)).mean()
                    logs["ent_coef_loss"].append(loss_aw.item())
                    self.ent_coef_optimizer.zero_grad()
                    loss_aw.backward()
                    self.ent_coef_optimizer.step()
                if self.k_ent_coef_optimizer is not None:
                    # increases a_k when H(pi_k) < rho log K'
                    loss_ak = (self.log_k_ent_coef * (aux["k_entropy"] - self.target_entropy_k)).mean()
                    logs["k_ent_coef_loss"].append(loss_ak.item())
                    self.k_ent_coef_optimizer.zero_grad()
                    loss_ak.backward()
                    self.k_ent_coef_optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        # ---- distill Q^W from Q ----
        for _ in range(self.noise_critic_grad_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            loss = self.update_noise_critic(data)
            logs["noise_critic_loss"].append(loss.item())
            self.critic_noise.optimizer.zero_grad()
            loss.backward()
            self.critic_noise.optimizer.step()

        self.critic_noise.set_training_mode(False)
        self._n_updates += gradient_steps
        if probs_acc:
            mean_probs = th.stack(probs_acc).mean(0).cpu().numpy()
            full = np.zeros(self.num_k)
            full[self.hybrid_actor.allowed_k.cpu().numpy()] = mean_probs
            self.last_mean_k_probs = full

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        stats = {f"train/{key}": float(np.mean(vals)) for key, vals in logs.items() if vals}
        for key, val in stats.items():
            self.logger.record(key, val)
        self.last_train_stats = stats

    def update_noise_critic(self, replay_data):
        obs = replay_data.observations
        B = obs.shape[0]
        allowed = self.hybrid_actor.allowed_k
        with th.no_grad():
            k = allowed[th.randint(0, allowed.numel(), (B,), device=allowed.device)]
            w = self._distill_w(B)
            a = self.steered_policy(obs, k, w).reshape(B, -1).to(self.device, th.float32)
            target = self.critic(obs, a)
        pred = self.critic_noise(obs, th.cat([self.hybrid_actor.onehot(k), self._scale_t(w)], dim=-1))
        return sum(0.5 * F.mse_loss(t.detach(), p) for t, p in zip(target, pred))

    # ---- acting ----
    def _record_k(self, k: th.Tensor) -> np.ndarray:
        k_np = k.detach().cpu().numpy()
        self.last_k = k_np
        return k_np

    def pop_k_counts(self) -> np.ndarray:
        """Histogram of k chosen in training rollouts since the last call."""
        c = self._k_counts.copy()
        self._k_counts[:] = 0
        return c

    def _act(self, observation, deterministic: bool) -> np.ndarray:
        obs = th.as_tensor(np.asarray(observation), device=self.device, dtype=th.float32)
        with th.no_grad():
            out = self.hybrid_actor.sample(obs, deterministic=deterministic)
            actions = self._decode(obs, out["k"], out["w"])
        self._record_k(out["k"])
        self.last_w = self._unscale_t(out["w"]).cpu().numpy()
        probs = np.zeros((obs.shape[0], self.num_k), dtype=np.float32)
        probs[:, self.hybrid_actor.allowed_k.cpu().numpy()] = out["probs"].cpu().numpy()
        self.last_probs = probs  # pi(k|s) over all K (zeros outside k_subset)
        return actions.cpu().numpy()

    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        assert self._last_obs is not None, "self._last_obs was not set"
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            obs = th.as_tensor(self._last_obs, device=self.device, dtype=th.float32)
            allowed = self.hybrid_actor.allowed_k
            k = allowed[th.randint(0, allowed.numel(), (n_envs,), device=allowed.device)]
            w_scaled = th.rand(n_envs, int(np.prod(self.action_space.shape)), device=self.device) * 2 - 1
            with th.no_grad():
                action = self._decode(obs, k, w_scaled).cpu().numpy()
            k_np = self._record_k(k)
        else:
            action = self._act(self._last_obs, deterministic=False)
            k_np = self.last_k
        self._k_counts += np.bincount(k_np, minlength=self.num_k)
        return action, action

    def predict_diffused(self, observation, state=None, episode_start=None, deterministic: bool = False):
        self.hybrid_actor.train(False)
        actions = self._act(observation, deterministic)
        return actions, None

    # ---- save ----
    def _get_torch_save_params(self):
        state_dicts, tensors = super()._get_torch_save_params()
        state_dicts = state_dicts + ["hybrid_actor", "hybrid_actor.optimizer", "critic_noise.optimizer"]
        if self.k_ent_coef_optimizer is not None:
            tensors = tensors + ["log_k_ent_coef"]
            state_dicts.append("k_ent_coef_optimizer")
        else:
            tensors = tensors + ["k_ent_coef_tensor"]
        return state_dicts, tensors + self._trust_region_tensors()

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["hybrid_actor"]  # noqa: RUF005
