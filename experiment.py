from __future__ import annotations

import logging
from agents import knowledgeAgent
from vi import Config, Window
from subjects import SubjectAgent
from environment import Environment
from logging_setup import setup_logging
import random
import math
from runtime_config import get_runtime_settings

logger = logging.getLogger(__name__)

def run_simulation(config: Config, settings: dict):
    env_width = settings["environment"]["width"]
    env_height = settings["environment"]["height"]
    context_length = settings["context"].get("p", 2)
    # Config-first: an ExperimentConfig (run_matrix.py) supplies learning_mode; a plain
    # vi.Config (single-run __main__) lacks it and falls back to yaml-derived settings.
    learning_mode = getattr(config, "learning_mode", None) or settings["learning_mode"]
    social_learning_enabled = learning_mode == "social"
    num_knowledge_agents = settings["agents"]["knowledge"]
    num_subject_agents = settings["agents"]["subjects"]
    ground_truth_snippets = list(settings["ground_truth"].get("snippets", []))

    if not ground_truth_snippets:
        raise ValueError("Active ground truth set must contain at least one snippet.")

    teleport_settings = settings.get("information_teleportation", {})
    teleport_enabled = teleport_settings.get("enabled", False)
    teleport_mode = teleport_settings.get("mode")

    is_pool_with_reemission = teleport_enabled and teleport_mode == "dynamic_pool_with_reemission"
    is_pool_without_reemission = teleport_enabled and teleport_mode == "dynamic_pool_without_reemission"

    if is_pool_with_reemission or is_pool_without_reemission:
        num_subject_agents = len(ground_truth_snippets)
        active_ratio = teleport_settings.get("active_ratio", 0.2)
        mean_swap_time = teleport_settings.get("mean_swap_time", 10.0)
        target_active = max(1, round(active_ratio * num_subject_agents))
        reemission_label = "with" if is_pool_with_reemission else "without"
        logger.info("Dynamic pool (%s reemission): %d total, %d initially active at %.0f%%, mean swap %.1fs",
                    reemission_label, num_subject_agents, target_active, active_ratio * 100, mean_swap_time)
    else:
        num_fragments = len(ground_truth_snippets)
        if num_subject_agents <= 0 or num_subject_agents > num_fragments:
            num_subject_agents = num_fragments
        ground_truth_snippets = ground_truth_snippets[:num_subject_agents]

    simulation = Environment(
        config=config,
        num_knowledge_agents=num_knowledge_agents,
        num_subject_agents=num_subject_agents,
    )

    num_snapshots = settings.get("num_snapshots", 30)
    snapshot_interval = settings.get("snapshot_interval_seconds", 10.0)
    
    logger.info("Simulation settings — environment: %dx%d, agents: %d knowledge / %d subjects, context: %d, social: %s, snapshots: %d x %.1fs, ground truth snippets: %d",
                env_width, env_height, num_knowledge_agents, num_subject_agents,
                context_length, social_learning_enabled,
                num_snapshots, snapshot_interval, len(ground_truth_snippets))

    def create_knowledge_agents(*args, **kwargs):
        return knowledgeAgent(context_size=context_length, social_learning_enabled=social_learning_enabled, *args, **kwargs)

    simulation.batch_spawn_agents(num_knowledge_agents, create_knowledge_agents, images=["images/robot.png"])

    grid_cols = math.ceil(math.sqrt(num_subject_agents))
    grid_rows = math.ceil(num_subject_agents / grid_cols)
    x_spacing = env_width / (grid_cols + 1)
    y_spacing = env_height / (grid_rows + 1)
    subject_positions = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            if len(subject_positions) >= num_subject_agents:
                break
            x = (col + 1) * x_spacing
            y = (row + 1) * y_spacing
            subject_positions.append((x, y))

    for fragment, position in zip(ground_truth_snippets, subject_positions):
        simulation.batch_spawn_agents(1, SubjectAgent, images=["images/villager.png"])
        subjects = [a for a in simulation._agents if getattr(a, "role", None) == "SUBJECT"]
        if subjects:
            subject = subjects[-1]
            subject.info = fragment
            subject.pos.update(position)

    if is_pool_with_reemission:
        simulation.initialize_dynamic_pool_with_reemission()

    if is_pool_without_reemission:
        simulation.initialize_dynamic_pool_without_reemission()
    
    return simulation

if __name__ == "__main__":
    simulation = None
    try:
        settings = get_runtime_settings()
        env_width = settings["environment"]["width"]
        fps = settings.get("fps")
        env_height = settings["environment"]["height"]
        simulation_config = Config(
            window=Window(env_width, env_height),
            seed=random.randint(0, 10),
            fps_limit=fps,
        )
        simulation = run_simulation(simulation_config, settings)
        setup_logging(simulation.run_dir / "run.log")
        simulation.run()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Simulation interrupted by user")
    finally:
        if simulation:
            simulation.stop()
            if not getattr(simulation, "_experiment_saved", False):
                simulation.save_experiment_data()
        
        
