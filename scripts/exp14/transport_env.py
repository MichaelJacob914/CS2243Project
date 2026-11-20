"""exp14 transport helpers: env construction, sim-state save/restore, chunk execution, phase labels.

Raw low-dim obs layout (exp13_headroom_analyze.py, checked there): robot0 eef_pos 0:3, eef_quat 3:7,
gripper_qpos 7:9, robot1 eef_pos 9:12, eef_quat 12:16, gripper_qpos 16:18, object 18:59 =
payload_pos +0:3, payload_quat +3:7, trash_pos +7:10, trash_quat +10:14, lid_handle_pos +14:17,
lid_handle_quat +17:21, target_bin_pos +21:24, trash_bin_pos +24:27, gripper0_to_payload +27:30,
gripper0_to_lid_handle +30:33, gripper1_to_payload +33:36, gripper1_to_trash +36:39,
payload_in_target_bin +39, trash_in_trash_bin +40.

Phases (plan numbering 1..5 -> codes 0..4; -1 = none of them, e.g. payload dropped / settled after success):
  0 before_contact  t > 0, lid / payload / trash all within 1 cm of their start, both grippers open
  1 open_grasp      lid moved or gripper0 closed, payload not yet carried (held by arm0 but within 5 cm of
                    start, or not held), trash sub-task not in progress
  2 trash           trash moved > 2 cm or held by gripper1, and trash not yet in the trash bin
  3 handover        payload held by arm0 and carried > 5 cm, or held by both arms
  4 place           payload held by arm1 alone, or in the target bin with gripper1 within 10 cm
Priority when several apply: place > handover > trash > open_grasp > before_contact.
"""

from __future__ import annotations

import numpy as np

import common

O = 18
LOW_DIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot1_eef_pos", "robot1_eef_quat",
                "robot1_gripper_qpos", "object"]  # fmt: skip
PHASES = ["before_contact", "open_grasp", "trash", "handover", "place"]
HOLD0, HOLD1, HOLD_TRASH, CLOSED = 0.10, 0.06, 0.06, 0.06
MAX_STEPS = 800


def feats(raw):
    raw = np.atleast_2d(raw)
    return {
        "payload": raw[:, O : O + 3], "trash": raw[:, O + 7 : O + 10], "lid": raw[:, O + 14 : O + 17],
        "g0_pay": np.linalg.norm(raw[:, O + 27 : O + 30], axis=1), "g0_lid": np.linalg.norm(raw[:, O + 30 : O + 33], axis=1),
        "g1_pay": np.linalg.norm(raw[:, O + 33 : O + 36], axis=1), "g1_trash": np.linalg.norm(raw[:, O + 36 : O + 39], axis=1),
        "pay_in_bin": raw[:, O + 39] > 0.5, "trash_in_bin": raw[:, O + 40] > 0.5,
        "g0_gap": raw[:, 7] - raw[:, 8], "g1_gap": raw[:, 16] - raw[:, 17],
        "eef0": raw[:, 0:3], "eef1": raw[:, 9:12],
    }  # fmt: skip


def phase_of(raw, raw0, t):
    f, f0 = feats(raw), feats(raw0)
    t = np.atleast_1d(t)
    d = lambda k: np.linalg.norm(f[k] - f0[k], axis=1)  # noqa: E731
    lid_mv, pay_mv, trash_mv = d("lid"), d("payload"), d("trash")
    g0c, g1c = f["g0_gap"] < CLOSED, f["g1_gap"] < CLOSED
    h0 = (f["g0_pay"] < HOLD0) & g0c
    h1 = (f["g1_pay"] < HOLD1) & g1c
    h1t = (f["g1_trash"] < HOLD_TRASH) & g1c
    ph = np.full(len(raw), -1)
    before = (t > 0) & (lid_mv < 0.01) & (pay_mv < 0.01) & (trash_mv < 0.01) & ~g0c & ~g1c
    trash_active = ((trash_mv > 0.02) | h1t) & ~f["trash_in_bin"]
    open_grasp = ((lid_mv > 0.01) | g0c) & ~(h0 & (pay_mv > 0.05)) & ~h1 & ~f["pay_in_bin"]
    handover = h0 & ((pay_mv > 0.05) | h1) & ~f["pay_in_bin"]
    place = (h1 & ~h0) | (f["pay_in_bin"] & (f["g1_pay"] < 0.10))
    ph[before] = 0
    ph[open_grasp] = 1
    ph[trash_active & (ph <= 1) & ~before] = 2
    ph[handover] = 3
    ph[place] = 4
    return ph


def raw_obs_from_dict(od):
    return np.concatenate([np.asarray(od[k], np.float64).ravel() for k in LOW_DIM_KEYS])


class TransportSim:
    """robomimic transport env with the policy's obs normalization; restore / step / chunk execution."""

    def __init__(self):
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from env_utils import make_robomimic_env

        self.w = make_robomimic_env(
            env="transport", normalization_path=common.NORM, low_dim_keys=LOW_DIM_KEYS,
            dppo_path=os.path.join(common.DSRL, "dppo"),
        )  # fmt: skip  (RobomimicLowdimWrapper around robomimic EnvRobosuite)
        self.renv = self.w.env  # robomimic EnvRobosuite
        self.rs = self.renv.env  # robosuite TwoArmTransport

    # --- state ---
    def reset_seeded(self, seed):
        np.random.seed(int(seed))  # robosuite's placement sampler draws from the global np RNG
        od = self.renv.reset()
        return self._obs(od)

    def get_state(self):
        return np.array(self.rs.sim.get_state().flatten())

    # Hidden state outside MuJoCo: robosuite's PandaGripper.format_action integrates
    # current_action += [-1, 1] * 0.01 * sign(cmd) (clipped to [-1, 1]) at every sim substep, and the gripper
    # actuators follow current_action. It must be saved / restored with the sim state.
    @property
    def n_substeps(self):
        return int(round(self.rs.control_timestep / self.rs.model_timestep))

    def get_grip(self):
        return np.concatenate([np.broadcast_to(np.asarray(r.gripper.current_action, np.float64), (2,)) for r in self.rs.robots])

    def set_grip(self, grip):
        grip = np.asarray(grip, np.float64).reshape(len(self.rs.robots), 2)
        for r, g in zip(self.rs.robots, grip):
            r.gripper.current_action = g.copy()

    def arm_qpos(self, state):
        """Arm joint positions (2 x 7) in a flattened MuJoCo state [time, qpos, qvel]."""
        st = np.asarray(state, np.float64)
        return np.concatenate([st[1 + np.asarray(r.controller.qpos_index)] for r in self.rs.robots])

    def restore(self, state, grip, init_joint=None):
        """Restore the MuJoCo state AND the hidden controller state:
        - the grippers' internal current_action;
        - the OSC nullspace target controller.initial_joint (set at every env reset from the reset qpos with random
          init noise), here = the episode's own t = 0 arm qpos (init_joint, 14 values) when given;
        - the OSC controller cache (eef pose, Jacobian, mass matrix): forced update, otherwise the first step after
          construction / a reset uses the reset pose as the base of the delta goal (review B1);
        - the solver warm-start (zeroed) and the observables' sampling timers (reset)."""
        self.renv.reset_to({"states": np.asarray(state, np.float64)})
        self.set_grip(grip)
        self.rs.sim.data.qacc_warmstart[:] = 0
        for k, r in enumerate(self.rs.robots):
            if init_joint is not None:
                r.controller.initial_joint = np.asarray(init_joint, np.float64)[7 * k : 7 * k + 7].copy()
            r.controller.update(force=True)
        for ob in self.rs._observables.values():  # observable sampling timers are hidden state too
            ob.reset()
        od = self.renv.get_observation()
        return self._obs(od)

    def _obs(self, od):
        raw = raw_obs_from_dict(od)
        return self.w.normalize_obs(raw).astype(np.float32), raw

    def norm(self, raw):
        return self.w.normalize_obs(raw).astype(np.float32)

    # --- stepping ---
    def step_chunk(self, chunk_norm):
        """Execute an (H, 14) normalized chunk. Returns (obs_norm, raw, success_any, n_steps)."""
        succ = False
        for a in chunk_norm:
            od, r, _, _ = self.renv.step(self.w.unnormalize_action(np.asarray(a, np.float64)))
            if r >= 1.0 - 1e-6:
                succ = True
        obs, raw = self._obs(od)
        return obs, raw, succ, len(chunk_norm)

    def is_success(self):
        return bool(self.renv.is_success()["task"])


def replay_grip_traj(actions_raw, n_substeps, speed=0.01):
    """Gripper current_action after each prefix of raw env actions (T, 14), starting from a reset (0):
    out[t] = state after actions[:t] (out[0] = 0). Per env step, n_substeps updates of
    clip(ca + [-1, 1] * speed * sign(cmd), -1, 1), exactly as PandaGripper.format_action."""
    acts = np.asarray(actions_raw)
    out = np.zeros((len(acts) + 1, 4))
    ca = np.zeros((2, 2))
    for t, a in enumerate(acts):
        for r, d in enumerate((6, 13)):
            step = np.array([-1.0, 1.0]) * speed * np.sign(a[d])
            for _ in range(n_substeps):
                ca[r] = np.clip(ca[r] + step, -1.0, 1.0)
        out[t + 1] = ca.reshape(-1)
    return out


def replay_grip(actions_raw, n_substeps, speed=0.01):
    return replay_grip_traj(actions_raw, n_substeps, speed)[-1]


def short_horizon(raw_before, raw_after):
    """Short-horizon effect of one chunk: object displacement (payload, lid handle, trash; 9-d) and, for each
    gripper, which object it approached most (index into [payload, lid_handle, trash] of the most negative
    change in eef-to-object distance)."""
    fb, fa = feats(raw_before), feats(raw_after)
    disp = np.concatenate([fa["payload"] - fb["payload"], fa["lid"] - fb["lid"], fa["trash"] - fb["trash"]], axis=1)
    out = {"disp": disp}
    for g, eef in (("toward0", "eef0"), ("toward1", "eef1")):
        dd = []
        for obj in ("payload", "lid", "trash"):
            db = np.linalg.norm(fb[obj] - fb[eef], axis=1)
            da = np.linalg.norm(fa[obj] - fa[eef], axis=1)
            dd.append(da - db)
        out[g] = np.argmin(np.stack(dd, 1), 1)
    out["eef_disp"] = np.concatenate([fa["eef0"] - fb["eef0"], fa["eef1"] - fb["eef1"]], axis=1)
    return out
