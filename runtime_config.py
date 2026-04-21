from functools import lru_cache
from typing import Dict, Any

from helpers import load_config
from constants import GROUND_TRUTH_LIBRARY


@lru_cache(maxsize=1)
def _get_config() -> Dict[str, Any]:
    return load_config() or {}


@lru_cache(maxsize=1)
def get_context_settings() -> Dict[str, Any]:
    config = _get_config()
    return config.get("context", {})


@lru_cache(maxsize=1)
def get_metric() -> str:
    config = _get_config()
    return config.get("metric", "cosine-bm25")


@lru_cache(maxsize=1)
def get_active_profile_key() -> str:
    config = _get_config()
    experiments = config.get("experiments", {})
    active_key = experiments.get("active_profile")
    profiles = experiments.get("profiles", {})
    if not active_key:
        raise ValueError("No active_profile set under experiments in configs.yaml")
    if active_key not in profiles:
        raise ValueError(f"Active profile '{active_key}' not defined under experiments.profiles")
    return active_key


@lru_cache(maxsize=1)
def get_active_profile() -> Dict[str, Any]:
    config = _get_config()
    experiments = config.get("experiments", {})
    profiles = experiments.get("profiles", {})
    active_key = get_active_profile_key()
    return profiles.get(active_key, {})


@lru_cache(maxsize=1)
def get_ground_truth_bundle() -> Dict[str, Any]:
    profile = get_active_profile()
    key = profile.get("ground_truth_key")
    if not key:
        raise ValueError("ground_truth_key must be defined in the active profile")
    if key not in GROUND_TRUTH_LIBRARY:
        raise ValueError(f"Unknown ground truth key '{key}'. Available keys: {list(GROUND_TRUTH_LIBRARY.keys())}")
    return GROUND_TRUTH_LIBRARY[key]


@lru_cache(maxsize=1)
def get_runtime_settings() -> Dict[str, Any]:
    config = _get_config()
    profile = get_active_profile()
    overrides = config.get("swarm_overrides") or {}

    ground_truth_bundle = get_ground_truth_bundle()
    num_subject_agents = profile.get("num_subject_agents") or len(ground_truth_bundle.get("snippets", []))

    environment_cfg = profile.get("environment") or {}
    width = environment_cfg.get("width")
    height = environment_cfg.get("height")
    if width is None or height is None:
        raise ValueError(f"Environment dimensions must be set for profile '{get_active_profile_key()}'")

    swarm_type = overrides.get("swarm_type") or profile.get("swarm_type") or "self_learning"
    social_override = overrides.get("social_learning_enabled")
    if social_override is None:
        social_learning_enabled = swarm_type == "social_learning"
    else:
        social_learning_enabled = bool(social_override)

    visualization_cfg = config.get("visualization") or {}
    live_plot_cfg = visualization_cfg.get("live_plot") or {}
    live_plot_enabled = live_plot_cfg.get("enabled")
    if live_plot_enabled is None:
        live_plot_enabled = True
    else:
        live_plot_enabled = bool(live_plot_enabled)

    update_interval_raw = live_plot_cfg.get("update_interval_ms")
    if update_interval_raw is None:
        live_plot_interval_ms = 3000
    else:
        try:
            live_plot_interval_ms = int(update_interval_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("visualization.live_plot.update_interval_ms must be an integer") from exc
        if live_plot_interval_ms <= 0:
            raise ValueError("visualization.live_plot.update_interval_ms must be positive")

    # Information teleportation settings (subject visibility / information decay)
    teleport_cfg = profile.get("information_teleportation") or {}
    teleport_enabled = teleport_cfg.get("enabled", False)
    teleport_mode = teleport_cfg.get("mode", "shuffle")  # "shuffle", "decay", or "dynamic_pool"
    visibility_prob = teleport_cfg.get("visibility_probability", 0.75)
    decay_count = teleport_cfg.get("decay_count", 3)  # legacy fixed-count decay (unused in continuous mode)
    decay_probability = teleport_cfg.get("decay_probability", 0.08)  # legacy per-subject leave chance (unused if lifetime_mean_time is set)
    interval_seconds = teleport_cfg.get("interval_seconds", 5.0)
    
    # Dynamic pool mode settings
    initial_active_count = teleport_cfg.get("initial_active_count", 5)  # subjects active at start
    appearance_mean_time = teleport_cfg.get("appearance_mean_time", 5.0)  # avg seconds until new subject appears
    lifetime_mean_time = teleport_cfg.get("lifetime_mean_time", 10.0)  # avg lifetime of active subject in seconds

    # Subject movement settings (controls whether subject agents move around)
    movement_cfg = profile.get("movement") or {}
    movement_enabled = bool(movement_cfg.get("enabled", False))
    movement_speed = movement_cfg.get("speed", 1.5)
    movement_angular_velocity = movement_cfg.get("angular_velocity", 5.0)

    experiments_cfg = config.get("experiments", {})
    num_snapshots = int(profile.get("num_snapshots") or experiments_cfg.get("num_snapshots", 60))
    snapshot_interval_seconds = float(profile.get("snapshot_interval_seconds") or experiments_cfg.get("snapshot_interval_seconds", 2.0))

    return {
        "profile_key": get_active_profile_key(),
        "num_snapshots": num_snapshots,
        "snapshot_interval_seconds": snapshot_interval_seconds,
        "swarm_type": swarm_type,
        "social_learning_enabled": social_learning_enabled,
        "environment": {
            "width": width,
            "height": height,
        },
        "agents": {
            "knowledge": profile.get("num_knowledge_agents", 1),
            "subjects": num_subject_agents,
        },
        "context": get_context_settings(),
        "metric": get_metric(),
        "ground_truth": ground_truth_bundle,
        "visualization": {
            "live_plot": {
                "enabled": live_plot_enabled,
                "update_interval_ms": live_plot_interval_ms,
            }
        },
        "movement": {
            "enabled": movement_enabled,
            "speed": movement_speed,
            "angular_velocity": movement_angular_velocity,
        },
        "information_teleportation": {
            "enabled": teleport_enabled,
            "mode": teleport_mode,  # "shuffle", "decay", or "dynamic_pool"
            "visibility_probability": visibility_prob,  # for shuffle mode
            "decay_count": decay_count,
            "decay_probability": decay_probability,
            "interval_seconds": interval_seconds,
            # dynamic_pool / continuous-decay mode settings
            "initial_active_count": initial_active_count,  # subjects active at start (dynamic_pool)
            "appearance_mean_time": appearance_mean_time,  # avg seconds until new subject appears (dynamic_pool)
            "lifetime_mean_time": lifetime_mean_time,  # avg lifetime of active subject (dynamic_pool / decay)
        },
    }

