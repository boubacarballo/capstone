from functools import lru_cache
from typing import Dict, Any

from helpers import load_config
from prompts import GROUND_TRUTH_LIBRARY


@lru_cache(maxsize=1)
def _get_config() -> Dict[str, Any]:
    return load_config() or {}


@lru_cache(maxsize=1)
def get_context_settings() -> Dict[str, Any]:
    config = _get_config()
    return config.get("context", {})


@lru_cache(maxsize=1)
def get_active_experiment() -> Dict[str, Any]:
    config = _get_config()
    active_key = config.get("active_experiment")
    experiments = config.get("experiments", {})
    defaults = config.get("defaults", {})

    if not active_key:
        raise ValueError("No active_experiment set in configs.yaml")
    if active_key not in experiments:
        raise ValueError(f"active_experiment '{active_key}' not defined under experiments in configs.yaml")

    experiment = experiments[active_key]

    merged = {**defaults, **experiment}
    merged["_key"] = active_key
    return merged


@lru_cache(maxsize=1)
def get_ground_truth_bundle() -> Dict[str, Any]:
    experiment = get_active_experiment()
    key = experiment.get("ground_truth_key")
    if not key:
        raise ValueError("ground_truth_key must be defined in defaults or the active experiment")
    if key not in GROUND_TRUTH_LIBRARY:
        raise ValueError(f"Unknown ground truth key '{key}'. Available keys: {list(GROUND_TRUTH_LIBRARY.keys())}")
    return GROUND_TRUTH_LIBRARY[key]


@lru_cache(maxsize=1)
def get_runtime_settings() -> Dict[str, Any]:
    config = _get_config()
    experiment = get_active_experiment()

    learning_mode = config.get("learning_mode", "self")
    if learning_mode not in ("self", "social"):
        raise ValueError(f"learning_mode must be 'self' or 'social', got '{learning_mode}'")
    social_learning_enabled = learning_mode == "social"
    swarm_type = "social_learning" if social_learning_enabled else "self_learning"

    ground_truth_bundle = get_ground_truth_bundle()
    num_subject_agents = experiment.get("num_subject_agents") or len(ground_truth_bundle.get("snippets", []))

    environment_cfg = experiment.get("environment") or {}
    width = environment_cfg.get("width")
    height = environment_cfg.get("height")
    if width is None or height is None:
        raise ValueError(f"Environment dimensions must be set in defaults or active experiment '{experiment['_key']}'")

    teleport_cfg = experiment.get("information_teleportation") or {}
    teleport_enabled = teleport_cfg.get("enabled", False)
    teleport_mode = teleport_cfg.get("mode")
    active_ratio = teleport_cfg.get("active_ratio", 0.2)
    mean_swap_time = teleport_cfg.get("mean_swap_time", 10.0)

    movement_cfg = experiment.get("movement") or {}
    movement_enabled = bool(movement_cfg.get("enabled", False))
    movement_speed = movement_cfg.get("speed", 1.5)
    movement_angular_velocity = movement_cfg.get("angular_velocity", 5.0)
    is_headless = config.get("is_headless", True)
    fps = config.get("fps", 60)

    num_snapshots = int(experiment.get("num_snapshots", 60))
    snapshot_interval_seconds = float(experiment.get("snapshot_interval_seconds", 2.0))

    return {
        "is_headless": is_headless,
        "fps": fps,
        "active_experiment": experiment["_key"],
        "learning_mode": learning_mode,
        "num_snapshots": num_snapshots,
        "snapshot_interval_seconds": snapshot_interval_seconds,
        "swarm_type": swarm_type,
        "social_learning_enabled": social_learning_enabled,
        "environment": {
            "width": width,
            "height": height,
        },
        "agents": {
            "knowledge": experiment.get("num_knowledge_agents", 1),
            "subjects": num_subject_agents,
        },
        "context": get_context_settings(),
        "ground_truth": ground_truth_bundle,
        "movement": {
            "enabled": movement_enabled,
            "speed": movement_speed,
            "angular_velocity": movement_angular_velocity,
        },
        "information_teleportation": {
            "enabled": teleport_enabled,
            "mode": teleport_mode,
            "active_ratio": active_ratio,
            "mean_swap_time": mean_swap_time,
        },
    }
