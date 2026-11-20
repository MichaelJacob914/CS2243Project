"""SAC agents over pi0 steering noise (jaxrl2, as in DSRL-pi0).

- Fixed k: DSRL-pi0's own `PixelSACLearner`, unchanged. The noise critic is Q(s, x_T, w); x_T is an obs key
  (dropped for k=0 with tiled w, where it never affects the action, so that case is exactly DSRL-pi0).
- Adaptive k: `HybridPixelSACLearner`, a hybrid SAC over (k, w), ported from adsrl/agent.py:DSRLAdaptiveK:
    actor: categorical pi(k|s,x_T) over the allowed k + one tanh-Gaussian head pi(w|s,x_T,k) per allowed k,
           on a shared trunk (own pixel encoder, as DSRL-pi0's actor).
    critic: Q(s, x_T, onehot(k), w), an ensemble like DSRL-pi0's (one-hot over all K steps).
    actor loss (exact sum over allowed k):
        sum_k pi(k) [a_w log pi(w_k|k) - Q(s,k,w_k)] + a_k sum_k pi(k) log pi(k)
    temperatures: a_w with DSRL-pi0's target entropy (-dim(w)/2 for 'auto'), a_k with target rho*log K'.
    k_entropy_bias='detach': pi(k) in the a_w log pi(w_k|k) term is stop-gradiented, so the categorical is trained
    only by Q and a_k H(pi_k) (no bonus for k whose w head is high-entropy); the heads' gradients are unchanged
    (each head still gets its own term weighted by pi(k)). 'keep' (default) is the loss above.
    critic target: (k', w') ~ pi(.|s'), y = r + gamma^H m Q_targ(s', k', w'). Like DSRL-pi0 (backup_entropy
    False), no entropy terms in the backup.
  Everything else (encoder, augmentation, ensemble size, reductions, optimizers) follows PixelSACLearner.
"""

from __future__ import annotations

import copy
import functools
import pathlib
from typing import Any, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints, train_state

from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.agents.pixel_sac.temperature import Temperature
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.constants import default_init, xavier_init
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.mlp import MLP
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.utils.target_update import soft_target_update

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class TrainState(train_state.TrainState):
    batch_stats: Any


# ------------------------------------------------------------------------------------------------
# networks
# ------------------------------------------------------------------------------------------------
class BroadcastPixelMultiplexer(PixelMultiplexer):
    """PixelMultiplexer (same parameters) that also accepts actions with an extra axis, (B, K', 1, A): the pixel
    encoder runs once per observation and its latent (and the other obs entries) are broadcast over K'. Used by
    the critic so the exact sum over k in the actor loss does not re-encode each image K' times."""

    @nn.compact
    def __call__(self, observations, actions=None, training: bool = False):
        observations = FrozenDict(observations)
        x = self.encoder(observations["pixels"], training)
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        x = observations.copy(add_or_replace={"pixels": x})
        if actions is not None and actions.ndim == 4:  # (B,K',1,A): jaxrl2's MLP flattens the last two axes
            kp = actions.shape[1]
            x = jax.tree.map(lambda v: jnp.broadcast_to(v[:, None], (v.shape[0], kp, *v.shape[1:])), x)
        if actions is None:
            return self.network(x, training=training)
        return self.network(x, actions, training=training)


class HybridPolicy(nn.Module):
    """Shared MLP trunk -> (log pi(k) [B,K'], mean [B,K',D], log_std [B,K',D])."""

    hidden_dims: Sequence[int]
    n_heads: int
    noise_dim: int
    freeze_k_uniform: bool = False

    @nn.compact
    def __call__(self, observations, training: bool = False):
        h = MLP(self.hidden_dims, activate_final=True)(observations, training=training)
        logits = nn.Dense(self.n_heads, kernel_init=default_init(1e-2))(h)
        mean = nn.Dense(self.n_heads * self.noise_dim, kernel_init=default_init(1e-2))(h)
        log_std = nn.Dense(self.n_heads * self.noise_dim, kernel_init=default_init(1e-2))(h)
        shape = (*h.shape[:-1], self.n_heads, self.noise_dim)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX).reshape(shape)
        if self.freeze_k_uniform:
            log_pk = jnp.full((*h.shape[:-1], self.n_heads), -jnp.log(self.n_heads))
        else:
            log_pk = jax.nn.log_softmax(logits, axis=-1)
        return log_pk, mean.reshape(shape), log_std


def squashed_sample(key, mean, log_std, magnitude: float, deterministic: bool = False):
    """w = magnitude * tanh(u), u ~ N(mean, std) (reparameterized). Returns (w, log pi(w)) summed over the last dim."""
    if deterministic:
        eps = jnp.zeros_like(mean)
    else:
        eps = jax.random.normal(key, mean.shape)
    u = mean + jnp.exp(log_std) * eps
    log_prob = (-0.5 * eps**2 - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
    # log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u)) (numerically stable, as distrax's Tanh bijector)
    log_prob = log_prob - (2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))).sum(-1)
    log_prob = log_prob - mean.shape[-1] * jnp.log(magnitude)
    return magnitude * jnp.tanh(u), log_prob


def augment_batch(rng, batch, color_jitter: bool, aug_next: bool):
    """Random crop (+ color jitter) of observation pixels, copied from PixelSACLearner._update_jit (1 camera)."""

    def aug(key, pixels):
        k1, k2 = jax.random.split(key)
        pixels = batched_random_crop(k1, pixels)
        if color_jitter:
            pixels = (color_transform(k2, pixels.astype(jnp.float32) / 255.0) * 255).astype(jnp.uint8)
        return pixels

    rng, k_obs, k_next = jax.random.split(rng, 3)
    obs = batch["observations"].copy(add_or_replace={"pixels": aug(k_obs, batch["observations"]["pixels"])})
    batch = batch.copy(add_or_replace={"observations": obs})
    if aug_next:
        nxt = batch["next_observations"].copy(
            add_or_replace={"pixels": aug(k_next, batch["next_observations"]["pixels"])}
        )
        batch = batch.copy(add_or_replace={"next_observations": nxt})
    return rng, batch


# ------------------------------------------------------------------------------------------------
# hybrid losses (pure functions; tested in tests/test_pi0_agent.py)
# ------------------------------------------------------------------------------------------------
def hybrid_actor_loss(actor_params, actor_apply, critic_params, critic_apply, obs, key, alpha_w, alpha_k,
                      allowed_k, num_k: int, magnitude: float, freeze_k_uniform: bool, critic_reduction: str,
                      detach_k_w_entropy: bool = False):
    log_pk, mean, log_std = actor_apply({"params": actor_params}, obs)
    B, Kp, D = mean.shape
    w, log_pw = squashed_sample(key, mean, log_std, magnitude)  # [B,K',D], [B,K']
    onehot = jax.nn.one_hot(jnp.broadcast_to(allowed_k, (B, Kp)), num_k)  # [B,K',K]
    kw = jnp.concatenate([onehot, w], -1)[:, :, None]  # [B,K',1,K+D]; critic encodes obs once, broadcasts over K'
    qs = critic_apply({"params": critic_params}, obs, kw)  # [n_qs, B, K']
    q = qs.min(0) if critic_reduction == "min" else qs.mean(0)
    probs = jnp.exp(log_pk)
    if detach_k_w_entropy:
        loss = (jax.lax.stop_gradient(probs) * alpha_w * log_pw - probs * q).sum(-1).mean()
    else:
        loss = (probs * (alpha_w * log_pw - q)).sum(-1).mean()
    if not freeze_k_uniform:
        loss = loss + alpha_k * (probs * log_pk).sum(-1).mean()
    info = {
        "actor_loss": loss,
        "entropy": -(probs * log_pw).sum(-1).mean(),  # E_k[-log pi(w_k|k)], drives a_w
        "k_entropy": -(probs * log_pk).sum(-1).mean(),  # H(pi_k), drives a_k
        "q_pi_in_actor": (probs * q).sum(-1).mean(),
        "k_probs": probs.mean(0),
        "std_pi_avg": jnp.exp(log_std).mean(),
        "mean_pi_norm": jnp.linalg.norm(mean, axis=-1).mean(),
    }
    return loss, info


def hybrid_sample(actor_apply, actor_params, obs, key, allowed_k, magnitude: float, deterministic: bool,
                  freeze_k_uniform: bool = False):
    """One (k, w) per row. Returns k (actual step), w [B,D], log pi(w|k) [B], probs [B,K']."""
    log_pk, mean, log_std = actor_apply({"params": actor_params}, obs)
    k_key, w_key = jax.random.split(key)
    if deterministic and not freeze_k_uniform:
        idx = jnp.argmax(log_pk, -1)
    else:
        idx = jax.random.categorical(k_key, log_pk, axis=-1)
    take = lambda x: jnp.take_along_axis(x, idx[:, None, None], axis=1)[:, 0]  # noqa: E731
    w, log_pw = squashed_sample(w_key, take(mean), take(log_std), magnitude, deterministic)
    return allowed_k[idx], w, log_pw, jnp.exp(log_pk)


@functools.partial(jax.jit, static_argnames=("num_k", "magnitude", "freeze_k_uniform", "critic_reduction",
                                             "color_jitter", "aug_next", "detach_k_w_entropy"))
def _hybrid_update(rng, actor, critic, target_critic_params, temp_w, temp_k, batch, allowed_k, target_entropy_w,
                   target_entropy_k, tau, num_k, magnitude, freeze_k_uniform, critic_reduction, color_jitter,
                   aug_next, detach_k_w_entropy=False):
    rng, batch = augment_batch(rng, batch, color_jitter, aug_next)
    alpha_w = temp_w.apply_fn({"params": temp_w.params})
    alpha_k = temp_k.apply_fn({"params": temp_k.params})
    B = batch["rewards"].shape[0]

    # ---- critic ----
    rng, key = jax.random.split(rng)
    nk, nw, _, _ = hybrid_sample(actor.apply_fn, actor.params, batch["next_observations"], key, allowed_k,
                                 magnitude, False, freeze_k_uniform)
    next_kw = jnp.concatenate([jax.nn.one_hot(nk, num_k), nw], -1)
    next_qs = critic.apply_fn({"params": target_critic_params}, batch["next_observations"], next_kw)
    next_q = next_qs.min(0) if critic_reduction == "min" else next_qs.mean(0)
    target_q = batch["rewards"] + batch["discount"] * batch["masks"] * next_q
    kw = jnp.concatenate([jax.nn.one_hot(batch["k"], num_k), batch["actions"].reshape(B, -1)], -1)

    def critic_loss_fn(params):
        qs = critic.apply_fn({"params": params}, batch["observations"], kw)
        loss = ((qs - target_q) ** 2).mean()
        return loss, {"critic_loss": loss, "q": qs.mean(), "target_q": target_q.mean()}

    grads, critic_info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    critic = critic.apply_gradients(grads=grads)
    target_critic_params = soft_target_update(critic.params, target_critic_params, tau)

    # ---- actor ----
    rng, key = jax.random.split(rng)
    grads, actor_info = jax.grad(hybrid_actor_loss, has_aux=True)(
        actor.params, actor.apply_fn, critic.params, critic.apply_fn, batch["observations"], key, alpha_w, alpha_k,
        allowed_k, num_k, magnitude, freeze_k_uniform, critic_reduction, detach_k_w_entropy)
    actor = actor.apply_gradients(grads=grads)

    # ---- temperatures (same form as jaxrl2 update_temperature) ----
    def temp_loss(params, apply_fn, entropy, target):
        t = apply_fn({"params": params})
        return t * (entropy - target)

    temp_w = temp_w.apply_gradients(grads=jax.grad(temp_loss)(
        temp_w.params, temp_w.apply_fn, actor_info["entropy"], target_entropy_w))
    if not freeze_k_uniform:
        temp_k = temp_k.apply_gradients(grads=jax.grad(temp_loss)(
            temp_k.params, temp_k.apply_fn, actor_info["k_entropy"], target_entropy_k))
    info = {**critic_info, **actor_info, "temperature": alpha_w, "k_temperature": alpha_k}
    return rng, actor, critic, target_critic_params, temp_w, temp_k, info


@functools.partial(jax.jit, static_argnames=("actor_apply", "magnitude", "deterministic", "freeze_k_uniform"))
def _hybrid_act(rng, actor_apply, actor_params, obs, allowed_k, magnitude, deterministic, freeze_k_uniform):
    rng, key = jax.random.split(rng)
    k, w, _, probs = hybrid_sample(actor_apply, actor_params, obs, key, allowed_k, magnitude, deterministic,
                                   freeze_k_uniform)
    return rng, k, w, probs


@functools.partial(jax.jit, static_argnames=("actor_apply",))
def _hybrid_heads(actor_apply, actor_params, obs):
    return actor_apply({"params": actor_params}, obs)


class HybridPixelSACLearner:
    def __init__(self, seed: int, observations, noise_dim: int, num_k: int, allowed_k: Sequence[int],
                 rho: float = 0.5, freeze_k_uniform: bool = False, actor_lr: float = 3e-4, critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4, hidden_dims: Sequence[int] = (256, 256),
                 cnn_features=(32, 32, 32, 32), cnn_strides=(2, 1, 1, 1), cnn_padding="VALID", latent_dim: int = 50,
                 discount: float = 0.99, tau: float = 0.005, critic_reduction: str = "mean",
                 dropout_rate=None, encoder_type="small", color_jitter=True, aug_next=True, use_bottleneck=True,
                 init_temperature: float = 1.0, num_qs: int = 2, target_entropy=None, action_magnitude: float = 1.0,
                 k_entropy_bias: str = "keep", **unused):
        assert encoder_type == "small", "only DSRL-pi0's 'small' encoder is wired up"
        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0],) * 3
        self.num_k = int(num_k)
        self.allowed_k = jnp.asarray(sorted({int(k) for k in allowed_k}), jnp.int32)
        self.n_heads = int(self.allowed_k.shape[0])
        self.noise_dim = int(noise_dim)
        self.magnitude = float(action_magnitude)
        self.freeze_k_uniform = bool(freeze_k_uniform)
        assert k_entropy_bias in ("keep", "detach"), k_entropy_bias
        self.k_entropy_bias = k_entropy_bias
        self.tau, self.discount, self.critic_reduction = tau, discount, critic_reduction
        self.color_jitter, self.aug_next = bool(color_jitter), bool(aug_next)
        self.action_chunk_shape = (1, self.noise_dim)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, tw_key, tk_key = jax.random.split(rng, 5)
        actor_def = PixelMultiplexer(encoder=Encoder(cnn_features, cnn_strides, cnn_padding),
                                     network=HybridPolicy(hidden_dims, self.n_heads, self.noise_dim,
                                                          self.freeze_k_uniform),
                                     latent_dim=latent_dim, use_bottleneck=use_bottleneck)
        actor_params = actor_def.init(actor_key, observations)["params"]
        self._actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params,
                                        tx=optax.adam(learning_rate=actor_lr), batch_stats=None)
        critic_def = BroadcastPixelMultiplexer(encoder=Encoder(cnn_features, cnn_strides, cnn_padding),
                                               network=StateActionEnsemble(hidden_dims, num_qs=num_qs),
                                      latent_dim=latent_dim, use_bottleneck=use_bottleneck)
        kw0 = jnp.zeros((1, self.num_k + self.noise_dim))
        critic_params = critic_def.init(critic_key, observations, kw0)["params"]
        self._critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params,
                                         tx=optax.adam(learning_rate=critic_lr), batch_stats=None)
        self._target_critic_params = copy.deepcopy(critic_params)
        temp_def = Temperature(init_temperature)
        self._temp = TrainState.create(apply_fn=temp_def.apply, params=temp_def.init(tw_key)["params"],
                                       tx=optax.adam(learning_rate=temp_lr), batch_stats=None)
        self._temp_k = TrainState.create(apply_fn=temp_def.apply, params=temp_def.init(tk_key)["params"],
                                         tx=optax.adam(learning_rate=temp_lr), batch_stats=None)
        self._rng = rng
        if target_entropy is None or target_entropy == "auto":
            self.target_entropy = -self.noise_dim / 2  # PixelSACLearner's 'auto'
        else:
            self.target_entropy = float(target_entropy)
        self.target_entropy_k = float(rho) * float(np.log(self.n_heads))

    def update(self, batch):
        (self._rng, self._actor, self._critic, self._target_critic_params, self._temp, self._temp_k,
         info) = _hybrid_update(self._rng, self._actor, self._critic, self._target_critic_params, self._temp,
                                self._temp_k, batch, self.allowed_k, self.target_entropy, self.target_entropy_k,
                                self.tau, self.num_k, self.magnitude, self.freeze_k_uniform, self.critic_reduction,
                                self.color_jitter, self.aug_next, self.k_entropy_bias == "detach")
        return info

    def head_log_std(self, obs):
        """Mean log_std of each allowed-k w head over obs and noise dims, (K',) numpy (no RNG use)."""
        _, _, log_std = _hybrid_heads(self._actor.apply_fn, self._actor.params, obs)
        return np.asarray(log_std).mean(axis=(0, 2))

    def act(self, obs, deterministic: bool = False):
        self._rng, k, w, probs = _hybrid_act(self._rng, self._actor.apply_fn, self._actor.params, obs,
                                             self.allowed_k, self.magnitude, deterministic, self.freeze_k_uniform)
        full = np.zeros((probs.shape[0], self.num_k), np.float32)
        full[:, np.asarray(self.allowed_k)] = np.asarray(probs)
        return np.asarray(k), np.asarray(w), full

    def get_rng(self):
        return self._rng

    def set_rng(self, rng):
        self._rng = rng

    @property
    def _save_dict(self):
        return {"critic": self._critic, "target_critic_params": self._target_critic_params, "actor": self._actor,
                "temp": self._temp, "temp_k": self._temp_k, "rng": self._rng}

    def restore_from(self, d):
        self._critic, self._target_critic_params = d["critic"], d["target_critic_params"]
        self._actor, self._temp, self._temp_k, self._rng = d["actor"], d["temp"], d["temp_k"], d["rng"]


class FixedKAgent:
    """DSRL-pi0's PixelSACLearner with a constant steering step k (act/save interface shared with the hybrid)."""

    def __init__(self, k: int, num_k: int, seed: int, observations, sample_action, **kwargs):
        self.k, self.num_k = int(k), int(num_k)
        self.learner = PixelSACLearner(seed, observations, sample_action, **kwargs)
        self.action_chunk_shape = self.learner.action_chunk_shape

    def update(self, batch):
        return self.learner.update(batch)

    def act(self, obs, deterministic: bool = False):
        w = self.learner.eval_actions(obs) if deterministic else self.learner.sample_actions(obs)
        w = np.asarray(w).reshape(w.shape[0], -1)
        B = w.shape[0]
        probs = np.zeros((B, self.num_k), np.float32)
        probs[:, self.k] = 1.0
        return np.full(B, self.k, np.int32), w, probs

    def get_rng(self):
        return self.learner._rng

    def set_rng(self, rng):
        self.learner._rng = rng

    @property
    def _save_dict(self):
        return {**self.learner._save_dict, "rng": self.learner._rng}

    def restore_from(self, d):
        L = self.learner
        L._critic, L._target_critic_params = d["critic"], d["target_critic_params"]
        L._actor, L._temp, L._rng = d["actor"], d["temp"], d["rng"]


def save_agent(agent, ckpt_dir: str, step: int):
    checkpoints.save_checkpoint(ckpt_dir, agent._save_dict, step, prefix="agent_", overwrite=True, keep=1)


def restore_agent(agent, ckpt_dir: str) -> Optional[int]:
    path = checkpoints.latest_checkpoint(ckpt_dir, prefix="agent_")
    if path is None:
        return None
    agent.restore_from(checkpoints.restore_checkpoint(path, agent._save_dict, prefix="agent_"))
    return int(pathlib.Path(path).name.split("_")[-1])
