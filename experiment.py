from __future__ import annotations

from agents import knowledgeAgent
from vi import Config, Window
from subjects import SubjectAgent
from environment import Environment
import random
import math
from runtime_config import get_runtime_settings

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
        print(f"🏊 Dynamic pool mode: {initial_active_count} initial subjects, {len(snippet_pool)} in pool")
    elif is_constant_ratio_pool:
        # Spawn ALL snippets; environment will control which fraction is visible
        num_fragments = len(ground_truth_snippets)
        num_subject_agents = num_fragments
        active_ratio = teleport_settings.get("active_ratio", 0.2)
        target_active = max(1, round(active_ratio * num_subject_agents))
        print(f"🎯 Constant ratio pool mode: {num_subject_agents} total subjects, "
              f"{target_active} active at {active_ratio:.0%} ratio")
    elif is_exponential_swap_pool:
        # Spawn ALL snippets; environment will control visibility via a single Poisson timer
        num_fragments = len(ground_truth_snippets)
        num_subject_agents = num_fragments
        active_ratio = teleport_settings.get("active_ratio", 0.2)
        mean_swap_time = teleport_settings.get("mean_swap_time", 10.0)
        target_active = max(1, round(active_ratio * num_subject_agents))
        print(f"Dynamic pool (with reemission) mode: {num_subject_agents} total subjects, "
              f"{target_active} initially active at {active_ratio:.0%} ratio, "
              f"mean swap interval {mean_swap_time}s")
    elif is_exponential_one_time_pool:
        # Spawn ALL snippets; each subject appears at most once (no re-appearances after swap-out)
        num_fragments = len(ground_truth_snippets)
        num_subject_agents = num_fragments
        active_ratio = teleport_settings.get("active_ratio", 0.2)
        mean_swap_time = teleport_settings.get("mean_swap_time", 10.0)
        target_active = max(1, round(active_ratio * num_subject_agents))
        print(f"Dynamic pool (without reemission) mode: {num_subject_agents} total subjects, "
              f"{target_active} initially active at {active_ratio:.0%} ratio, "
              f"mean swap interval {mean_swap_time}s (no re-appearances)")
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
    
    print(f"Running the simulation with the following settings")
    print(f"Environment: {env_width}x{env_height}")
    print(f"Agents: {num_knowledge_agents} knowledge, {num_subject_agents} subjects")
    print(f"Context length: {context_length}")
    print(f"Social learning enabled: {social_learning_enabled}")
    print(f"Snapshots: {num_snapshots} × {snapshot_interval}s = {num_snapshots * snapshot_interval}s total")
    print(f"Ground truth snippets: {len(ground_truth_snippets)}")

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
        simulation.run()
    except KeyboardInterrupt:
        print("\n⏹️  Simulation interrupted by user")
    finally:
        if simulation:
            simulation.stop()
            if not getattr(simulation, "_experiment_saved", False):
                simulation.save_experiment_data()
        
        
