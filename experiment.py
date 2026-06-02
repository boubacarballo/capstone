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

def run_simulation():
    settings = get_runtime_settings()
    env_width = settings["environment"]["width"]
    env_height = settings["environment"]["height"]
    context_length = settings["context"].get("p", 2)
    social_learning_enabled = settings["social_learning_enabled"]
    num_knowledge_agents = settings["agents"]["knowledge"]
    num_subject_agents = settings["agents"]["subjects"]
    ground_truth_snippets = list(settings["ground_truth"].get("snippets", []))

    if not ground_truth_snippets:
        raise ValueError("Active ground truth set must contain at least one snippet.")

    # Information teleportation settings
    teleport_settings = settings.get("information_teleportation", {})
    teleport_enabled = teleport_settings.get("enabled", False)
    teleport_mode = teleport_settings.get("mode", "shuffle")
    initial_active_count = teleport_settings.get("initial_active_count", 5)
    
    # For dynamic_pool mode: shuffle and split snippets
    is_dynamic_pool = teleport_enabled and teleport_mode == "dynamic_pool"
    is_constant_ratio_pool = teleport_enabled and teleport_mode == "constant_ratio_pool"
    is_exponential_swap_pool = teleport_enabled and teleport_mode == "dynamic_pool_with_reemission"
    is_exponential_one_time_pool = teleport_enabled and teleport_mode == "dynamic_pool_without_reemission"
    snippet_pool = []
    
    if is_dynamic_pool:
        # Shuffle all snippets
        random.shuffle(ground_truth_snippets)
        # Split into initial active and pool
        initial_active_count = min(initial_active_count, len(ground_truth_snippets))
        initial_snippets = ground_truth_snippets[:initial_active_count]
        snippet_pool = ground_truth_snippets[initial_active_count:]
        ground_truth_snippets = initial_snippets
        num_subject_agents = initial_active_count
        logger.info("Dynamic pool mode: %d initial subjects, %d in pool", initial_active_count, len(snippet_pool))
    elif is_constant_ratio_pool:
        # Spawn ALL snippets; environment will control which fraction is visible
        num_fragments = len(ground_truth_snippets)
        num_subject_agents = num_fragments
        active_ratio = teleport_settings.get("active_ratio", 0.2)
        target_active = max(1, round(active_ratio * num_subject_agents))
        logger.info("Constant ratio pool mode: %d total subjects, %d active at %.0f%% ratio",
                    num_subject_agents, target_active, active_ratio * 100)
    elif is_exponential_swap_pool:
        # Spawn ALL snippets; environment will control visibility via a single Poisson timer
        num_fragments = len(ground_truth_snippets)
        num_subject_agents = num_fragments
        active_ratio = teleport_settings.get("active_ratio", 0.2)
        mean_swap_time = teleport_settings.get("mean_swap_time", 10.0)
        target_active = max(1, round(active_ratio * num_subject_agents))
        logger.info("Dynamic pool (with reemission) mode: %d total subjects, %d initially active at %.0f%% ratio, mean swap interval %.1fs",
                    num_subject_agents, target_active, active_ratio * 100, mean_swap_time)
    elif is_exponential_one_time_pool:
        # Spawn ALL snippets; each subject appears at most once (no re-appearances after swap-out)
        num_fragments = len(ground_truth_snippets)
        num_subject_agents = num_fragments
        active_ratio = teleport_settings.get("active_ratio", 0.2)
        mean_swap_time = teleport_settings.get("mean_swap_time", 10.0)
        target_active = max(1, round(active_ratio * num_subject_agents))
        logger.info("Dynamic pool (without reemission) mode: %d total subjects, %d initially active at %.0f%% ratio, mean swap interval %.1fs (no re-appearances)",
                    num_subject_agents, target_active, active_ratio * 100, mean_swap_time)
    else:
        # Standard mode: align subject agent count with available snippets
        num_fragments = len(ground_truth_snippets)
        if num_subject_agents <= 0 or num_subject_agents > num_fragments:
            num_subject_agents = num_fragments
        ground_truth_snippets = ground_truth_snippets[:num_subject_agents]

    simulation_config = Config(window=Window(env_width, env_height), seed=random.randint(0, 10))
    simulation = Environment(
        config=simulation_config,
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

    # Initialize dynamic pool if in dynamic_pool mode
    if is_dynamic_pool and snippet_pool:
        simulation.initialize_dynamic_pool(snippet_pool)
    
    # Initialize constant ratio pool if in constant_ratio_pool mode
    if is_constant_ratio_pool:
        simulation.initialize_constant_ratio_pool()

    if is_exponential_swap_pool:
        simulation.initialize_dynamic_pool_with_reemission()

    if is_exponential_one_time_pool:
        simulation.initialize_dynamic_pool_without_reemission()
    
    return simulation

# Create and run the story environment
if __name__ == "__main__":
    simulation = None
    try:
        simulation = run_simulation()
        setup_logging(simulation.run_dir / "run.log")
        simulation.run()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Simulation interrupted by user")
    finally:
        if simulation:
            simulation.stop()
            if not getattr(simulation, "_experiment_saved", False):
                simulation.save_experiment_data()
        
        
