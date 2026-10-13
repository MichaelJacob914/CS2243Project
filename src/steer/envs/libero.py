"""LIBERO task wrapper reproducing openpi's official eval protocol (third_party/openpi/examples/libero/main.py):
fixed initial states, 10 settling steps, 180-degree image rotation, 224 px padded resize, per-suite step limits."""

import math
import pathlib

import numpy as np

DUMMY_ACTION = [0.0] * 6 + [-1.0]
RESOLUTION = 256  # render resolution of the training data
MAX_STEPS = {  # longest training demo + margin, from openpi
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def suite_size(suite: str) -> int:
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[suite]().n_tasks


class LiberoTask:
    fps = 20  # LIBERO control rate

    def __init__(self, suite: str, task_id: int, seed: int = 7, resize: int = 224, num_steps_wait: int = 10):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task_suite = benchmark.get_benchmark_dict()[suite]()
        task = task_suite.get_task(task_id)
        self.prompt = str(task.language)
        self.init_states = task_suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self._env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RESOLUTION, camera_widths=RESOLUTION)
        self._env.seed(seed)  # openpi: the seed moves objects even with fixed initial states
        self.max_steps = MAX_STEPS[suite]
        self._resize, self._wait = resize, num_steps_wait
        self._obs = self._img = None

    def reset(self, episode: int) -> dict:
        self._env.reset()
        obs = self._env.set_init_state(self.init_states[episode % len(self.init_states)])
        for _ in range(self._wait):  # objects are dropped into the scene; let them settle
            obs, _, _, _ = self._env.step(DUMMY_ACTION)
        return self._policy_obs(obs)

    def step(self, action: np.ndarray) -> tuple[bool, bool, dict]:
        obs, _, done, _ = self._env.step(np.asarray(action).tolist())
        return done, done, self._policy_obs(obs)  # LIBERO ends an episode only on success

    def frame(self) -> np.ndarray:
        return self._img

    def close(self) -> None:
        self._env.close()

    def _policy_obs(self, obs: dict) -> dict:
        from openpi_client import image_tools

        def prep(x):
            x = np.ascontiguousarray(x[::-1, ::-1])  # rotate 180 degrees to match training preprocessing
            return image_tools.convert_to_uint8(image_tools.resize_with_pad(x, self._resize, self._resize))

        self._img = prep(obs["agentview_image"])
        return {
            "observation/image": self._img,
            "observation/wrist_image": prep(obs["robot0_eye_in_hand_image"]),
            "observation/state": np.concatenate(
                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
            "prompt": self.prompt,
        }


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite's quat2axisangle (xyzw), as copied into openpi's LIBERO example."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
