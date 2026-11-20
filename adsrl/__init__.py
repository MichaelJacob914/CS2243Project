"""Adaptive-step DSRL: steer a frozen DPPO diffusion policy at a chosen DDIM step."""

import os
import sys

# dppo modules import each other as top-level `model.*`, so put dsrl/dppo on the path.
_DPPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dsrl", "dppo")
if os.path.isdir(_DPPO_DIR) and _DPPO_DIR not in sys.path:
    sys.path.append(_DPPO_DIR)

from adsrl.sampler import ddim_step, num_ddim_steps, steered_ddim_sample  # noqa: E402
from adsrl.base_policy import SteeredBasePolicy  # noqa: E402

__all__ = ["steered_ddim_sample", "ddim_step", "num_ddim_steps", "SteeredBasePolicy"]
