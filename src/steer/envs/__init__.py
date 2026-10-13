"""Simulator wrappers. Each task exposes: prompt, max_steps, fps, reset(episode) -> policy obs,
step(action) -> (done, success, policy obs), frame() -> HxWx3 uint8, close()."""


def make_task(env: str, **kwargs):
    if env == "libero":
        from steer.envs.libero import LiberoTask

        return LiberoTask(**kwargs)
    if env == "aloha":
        from steer.envs.aloha import AlohaTask

        return AlohaTask(**kwargs)
    raise ValueError(f"unknown env {env!r} (libero|aloha)")
