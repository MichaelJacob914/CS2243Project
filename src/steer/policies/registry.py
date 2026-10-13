"""Every policy we can run, by name. Adding a model = one entry here + a config in configs/eval/."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    config: str  # openpi training-config name (third_party/openpi/src/openpi/training/config.py)
    checkpoint: str  # gs:// path; resolved through the $OPENPI_DATA_HOME cache
    env: str  # "libero" | "aloha"
    replan_steps: int  # actions executed per model call (default; configs may override)


POLICIES: dict[str, PolicySpec] = {
    # Chunk of 50 actions; openpi's LIBERO eval executes 5 before re-querying.
    "pi0_libero": PolicySpec("pi0_libero", "gs://openpi-assets/checkpoints/pi0_libero", "libero", 5),
    # openpi's ALOHA-sim example executes 10 actions per query.
    "pi0_aloha_sim": PolicySpec("pi0_aloha_sim", "gs://openpi-assets/checkpoints/pi0_aloha_sim", "aloha", 10),
    # pi0.5: chunk of 10, same LIBERO inputs. Needs `scripts/download_ckpts.py pi05_libero` first.
    "pi05_libero": PolicySpec("pi05_libero", "gs://openpi-assets/checkpoints/pi05_libero", "libero", 5),
}


def get(name: str) -> PolicySpec:
    try:
        return POLICIES[name]
    except KeyError:
        raise KeyError(f"unknown policy {name!r}; known: {sorted(POLICIES)}") from None
