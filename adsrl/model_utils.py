"""Helpers to build DPPO diffusion models from the dsrl task configs."""

from __future__ import annotations

import os

import hydra
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSRL_DIR = os.path.join(PROJECT_ROOT, "dsrl")

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


def load_task_cfg(task: str, cfg_path: str | None = None):
    """Load dsrl/cfg/robomimic/dsrl_<task>.yaml. Relative paths in it are relative to dsrl/."""
    cfg_path = cfg_path or os.path.join(DSRL_DIR, "cfg", "robomimic", f"dsrl_{task}.yaml")
    return OmegaConf.load(cfg_path)


def abs_dsrl_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(DSRL_DIR, p))


def build_model(cfg, device: str = "cpu", ddim_steps: int | None = None, network_path: str | None = None):
    """Instantiate cfg.model (a DiffusionEval) on `device`, optionally overriding K and the checkpoint."""
    import adsrl  # noqa: F401  (puts dsrl/dppo on sys.path)

    cfg = cfg.copy()
    cfg.device = device
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["device"] = device
    model_cfg["network_path"] = abs_dsrl_path(network_path or model_cfg["network_path"])
    if ddim_steps is not None:
        model_cfg["ddim_steps"] = int(ddim_steps)
    model = hydra.utils.instantiate(OmegaConf.create(model_cfg))
    return model.eval()
