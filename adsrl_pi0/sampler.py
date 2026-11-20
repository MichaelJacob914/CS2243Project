"""Steered pi0 sampling: run openpi's Euler flow sampler, re-noising the clean estimate at step k.

openpi convention (models/pi0.py): t=1 is noise, t=0 is data, x_t = t*eps + (1-t)*a, and the network
predicts the velocity v = eps - a. Euler with N steps, dt = -1/N, step j = 0..N-1 at time t_j = 1 + j*dt
(accumulated in float32 exactly as openpi does: time <- time + dt):

    x_{j+1} = x_j + dt * v_j

From one velocity evaluation at (x_j, t_j):
    clean estimate  a_hat_j   = x_j - t_j * v_j
    implied noise   eps_hat_j = x_j + (1 - t_j) * v_j

Steering at k (k = 0..N-1): the input to step k is re-noised with the agent's w at the noise level t_k:
    k = 0 : x_0 = w                                          (x_T replaced: standard DSRL)
    k > 0 : x_k = t_k * w + (1 - t_k) * a_hat_{k-1}

Because the Euler update is affine in the same interpolant, x_{j+1} = t_{j+1} eps_hat_j + (1 - t_{j+1}) a_hat_j
exactly (substitute and use t_{j+1} - t_j = dt). So w = eps_hat_{k-1} reproduces the unsteered sample, which is
the flow-matching analogue of DDIM's "replace eps_hat with w". x_T is the noise for steps 0..k-1.

The loop is a `lax.while_loop` with the same carry and stopping rule as openpi, so k=0 matches
`Pi0.sample_actions(noise=w)`. k is per sample, so a batch can mix different steering steps.

Model-agnostic over pi0 and pi0.5: both share this loop and the KV-cached prefix; they differ only inside the
velocity call. pi0 (dsrl_pi0/openpi fork) embeds a continuous state token in the suffix and mixes the time embedding
into the action tokens with an MLP; `embed_suffix` returns 3 values. pi0.5 (upstream openpi, `Pi0Config(pi05=True)`)
has no state token in the suffix (state, if used, is tokenized into the prompt by the input transforms) and conditions
the action expert on time through adaRMSNorm; upstream `embed_suffix` returns a 4th value, the adaRMS conditioning,
which is passed to the LLM as `adarms_cond=[None, cond]` (None for pi0 under upstream). `_velocity` handles both.
"""

from __future__ import annotations

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

NUM_STEPS = 10  # openpi default for pi0 and pi0.5

# --model name -> (openpi config name, checkpoint). pi0 needs .venv-pi0 (dsrl_pi0/openpi fork); pi05 needs
# .venv-pi05 (upstream openpi, third_party/openpi_upstream). Checkpoints are cached under OPENPI_DATA_HOME.
MODELS = {
    "pi0": ("pi0_libero", "s3://openpi-assets/checkpoints/pi0_libero"),
    "pi05": ("pi05_libero", "gs://openpi-assets/checkpoints/pi05_libero"),
}


def _velocity(model, observation, prefix_mask, kv_cache, x_t, time):
    """v(x_t, t) given the prefix KV cache: openpi's per-step suffix forward pass, for pi0 and pi0.5."""
    from openpi.models.pi0 import make_attn_mask

    batch_size = x_t.shape[0]
    out = model.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
    suffix_tokens, suffix_mask, suffix_ar_mask = out[:3]
    extra = {"adarms_cond": [None, out[3]]} if len(out) == 4 else {}  # upstream openpi (pi0.5 adaRMS time cond)
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
    pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens], mask=full_attn_mask, positions=pos, kv_cache=kv_cache, **extra
    )
    assert prefix_out is None
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])


def steered_sample_actions(model, observation, x_T, k, w, num_steps: int = NUM_STEPS, return_trajectory: bool = False):
    """Steered version of openpi `Pi0.sample_actions`.

    Args:
        model: openpi `Pi0` nnx module.
        observation: openpi `Observation` (batched, already through the policy input transforms).
        x_T: (B, H, A) initial noise. w: (B, H, A) steering noise. k: (B,) int steering step in [0, num_steps-1].
        return_trajectory: also return the input state of every step, (num_steps + 1, B, H, A) with entry j the
            state at t_j (entry k is the re-noised state) and the last entry the output (for tests).
    Returns:
        (B, H, A) actions in normalized model space [, trajectory].
    """
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    k = jnp.asarray(k, jnp.int32).reshape(batch_size, 1, 1)
    x_T = jnp.asarray(x_T, jnp.float32)
    w = jnp.asarray(w, jnp.float32)

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def velocity(x_t, time):
        return _velocity(model, observation, prefix_mask, kv_cache, x_t, time)

    def step(carry):
        x_t, time, j = carry[:3]
        v_t = velocity(x_t, time)
        x_next = x_t + dt * v_t
        time_next = time + dt
        # re-noise the clean estimate at the next step's noise level when k == j + 1
        a_hat = x_t - time * v_t
        x_steer = time_next * w + (1.0 - time_next) * a_hat
        x_next = jnp.where(k == j + 1, x_steer, x_next)
        if return_trajectory:
            return x_next, time_next, j + 1, carry[3].at[j + 1].set(x_next)
        return x_next, time_next, j + 1

    def cond(carry):
        return carry[1] >= -dt / 2

    x_init = jnp.where(k == 0, w, x_T)
    if return_trajectory:
        traj = jnp.zeros((num_steps + 1, *x_init.shape), x_init.dtype).at[0].set(x_init)
        x_0, _, _, traj = jax.lax.while_loop(cond, step, (x_init, 1.0, jnp.int32(0), traj))
        return x_0, traj
    x_0, _, _ = jax.lax.while_loop(cond, step, (x_init, 1.0, jnp.int32(0)))
    return x_0


def plain_features(model, observation, x_T, k: int, num_steps: int = NUM_STEPS):
    """Plan feature for steering at a fixed step k (static), from the plain (unsteered) trajectory from x_T:
    k = 0: the plain sample (openpi `sample_actions(noise=x_T)`, same while_loop);
    k > 0: a_hat_{k-1} = x_{k-1} - t_{k-1} v_{k-1}, the clean estimate `steered_sample_actions` re-noises at k.
    The "prefix" half of a steered sample: the agent picks w from it, then `steered_sample_actions` (which recomputes
    the k plain steps) is the suffix. x_T (B, H, A) -> (B, H, A) in normalized model space."""
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    x_T = jnp.asarray(x_T, jnp.float32)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def velocity(x_t, time):
        return _velocity(model, observation, prefix_mask, kv_cache, x_t, time)

    def step(carry):
        x_t, time = carry
        return x_t + dt * velocity(x_t, time), time + dt

    if k == 0:
        x_0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (x_T, 1.0))
        return x_0
    x, time = jax.lax.fori_loop(0, k - 1, lambda j, c: step(c), (x_T, jnp.float32(1.0)))
    return x - time * velocity(x, time)


def implied_noise_at(model, observation, x_T, k_target: int, num_steps: int = NUM_STEPS):
    """eps_hat_{k-1} along the unsteered trajectory from x_T: the w that makes steering at k a no-op.
    Test helper (runs the plain sampler for k_target-1 steps, eager python loop)."""
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    assert k_target >= 1
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    _, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=make_attn_mask(prefix_mask, prefix_ar_mask), positions=jnp.cumsum(prefix_mask, 1) - 1
    )
    x, time = jnp.asarray(x_T, jnp.float32), jnp.float32(1.0)
    for j in range(k_target):
        v = _velocity(model, observation, prefix_mask, kv_cache, x, time)
        if j == k_target - 1:
            return x + (1.0 - time) * v
        x, time = x + dt * v, time + dt
    raise AssertionError


def module_jit_fn(fn, model, **jit_kwargs):
    """jit `fn(model, *args)` with the nnx module's state frozen (like openpi's nnx_utils.module_jit)."""
    graphdef, state = nnx.split(model)

    def f(state, *args, **kwargs):
        return fn(nnx.merge(graphdef, state), *args, **kwargs)

    jitted = jax.jit(f, **jit_kwargs)
    return lambda *args, **kwargs: jitted(state, *args, **kwargs)


class SteeredPi0:
    """pi0 / pi0.5 LIBERO policy with per-sample steering. Wraps an openpi `Policy` for its input/output transforms.

    `infer(obs_list, x_T, k, w)`: obs_list is a list of B openpi LIBERO input dicts; x_T and w are (B, H, A) or
    (B, 1, A) (a single noise row tiled over the horizon, the DSRL-pi0 convention); k is (B,) ints.
    Returns (B, H, 7) env actions.
    """

    def __init__(self, policy, model, num_steps: int = NUM_STEPS):
        self.policy = policy
        self.model = model
        self.num_steps = num_steps
        self.horizon = model.action_horizon
        self.action_dim = model.action_dim
        self._sample = module_jit_fn(steered_sample_actions, model, static_argnames=("num_steps",))
        self._features = module_jit_fn(plain_features, model, static_argnames=("k", "num_steps"))

    @classmethod
    def load_model(cls, name: str = "pi0"):
        """Load a LIBERO policy by --model name (MODELS)."""
        return cls.load(*MODELS[name])

    @classmethod
    def load(cls, config_name: str = "pi0_libero", checkpoint: str = "s3://openpi-assets/checkpoints/pi0_libero"):
        """Same as openpi `policy_config.create_trained_policy`, but keeps a handle on the model."""
        import openpi.transforms as transforms
        from openpi.models import model as _model
        from openpi.policies import policy as _policy
        from openpi.shared import download
        from openpi.shared import normalize as _normalize
        from openpi.training import config as openpi_config

        config = openpi_config.get_config(config_name)
        # openpi-assets is public: anonymous GCS access (only used when the checkpoint is not cached yet)
        ckpt = download.maybe_download(checkpoint, **({"token": "anon"} if checkpoint.startswith("gs://") else {}))
        model = config.model.load(_model.restore_params(ckpt / "params", dtype=jnp.bfloat16))
        data_config = config.data.create(config.assets_dirs, config.model)
        # = openpi.training.checkpoints.load_norm_stats, which upstream can't import without lerobot (dropped in .venv-pi05)
        norm_stats = _normalize.load(ckpt / "assets" / data_config.asset_id)
        policy = _policy.Policy(
            model,
            transforms=[
                transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ],
            output_transforms=[
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ],
            metadata=config.policy_metadata,
        )
        return cls(policy, model)

    def tile(self, x):
        x = np.asarray(x, np.float32)
        if x.ndim == 2:
            x = x[:, None, :]
        if x.shape[1] == 1:
            x = np.repeat(x, self.horizon, axis=1)
        assert x.shape[1:] == (self.horizon, self.action_dim), x.shape
        return x

    def prepare(self, obs_list):
        from openpi.models import model as _model

        inputs = [self.policy._input_transform(jax.tree.map(lambda x: x, o)) for o in obs_list]
        batched = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *inputs)
        return _model.Observation.from_dict(batched), batched["state"]

    def sample_normalized(self, obs_list, x_T, k, w):
        """Pads the batch (to 1, 2, 4, then multiples of 4) to limit recompiles; returns the first B rows."""
        b = len(obs_list)
        bp = (1 << (b - 1).bit_length()) if b <= 4 else -(-b // 4) * 4
        pad = lambda x: np.concatenate([x, np.repeat(x[:1], bp - b, axis=0)]) if bp > b else x  # noqa: E731
        observation, state = self.prepare(list(obs_list) + [obs_list[0]] * (bp - b))
        k = pad(np.broadcast_to(np.asarray(k, np.int32), (b,)).copy())
        out = self._sample(observation, pad(self.tile(x_T)), k, pad(self.tile(w)), num_steps=self.num_steps)
        return out[:b], state[:b]

    def plan_feature(self, obs_list, x_T, k: int, n_real: int = 7):
        """`plain_features` for fixed k: (B, H * n_real) float32, the real action channels in normalized model space
        (before Unnormalize). Same batch padding as sample_normalized."""
        b = len(obs_list)
        bp = (1 << (b - 1).bit_length()) if b <= 4 else -(-b // 4) * 4
        x_T = self.tile(x_T)
        x_T = np.concatenate([x_T, np.repeat(x_T[:1], bp - b, axis=0)]) if bp > b else x_T
        observation, _ = self.prepare(list(obs_list) + [obs_list[0]] * (bp - b))
        out = self._features(observation, x_T, k=int(k), num_steps=self.num_steps)
        return np.asarray(out[:b, :, :n_real], np.float32).reshape(b, -1)

    def infer(self, obs_list, x_T, k, w):
        actions, state = self.sample_normalized(obs_list, x_T, k, w)
        out = self.policy._output_transform({"state": np.asarray(state), "actions": np.asarray(actions)})
        return np.asarray(out["actions"])

