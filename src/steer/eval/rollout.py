"""One episode: query the policy for a chunk, execute `replan_steps` actions, repeat."""

import collections

import numpy as np


def run_episode(policy, task, sampler, episode: int, replan_steps: int, record: bool = False) -> tuple[dict, list]:
    obs = task.reset(episode)
    plan: collections.deque = collections.deque()
    frames, infer_ms = [], []
    success, steps, error = False, 0, None
    try:
        while steps < task.max_steps:
            if not plan:
                chunk = policy.act(obs, sampler(obs))
                infer_ms.append(policy.last_infer_ms)
                if len(chunk) < replan_steps:
                    raise ValueError(f"policy returned {len(chunk)} actions, replan_steps={replan_steps}")
                plan.extend(chunk[:replan_steps])
            if record:
                frames.append(task.frame())
            done, success, obs = task.step(plan.popleft())
            steps += 1
            if done:
                break
    except Exception as e:  # noqa: BLE001 -- openpi's eval also counts a crashed episode as a failure
        error = repr(e)
    return {
        "success": bool(success),
        "steps": steps,
        "queries": len(infer_ms),
        "infer_ms": float(np.mean(infer_ms)) if infer_ms else None,
        "error": error,
    }, frames
