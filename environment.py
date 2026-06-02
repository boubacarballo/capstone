from __future__ import annotations

import logging
from vi import Agent, Config, Simulation, Window, HeadlessSimulation
import json
from llm import LLM
import pygame as pg
from pathlib import Path
from runtime_config import get_runtime_settings
from datetime import datetime
import random

logger = logging.getLogger(__name__)


class Environment(Simulation):
    def __init__(self, config=None, num_knowledge_agents: int = 2, num_subject_agents: int = 5):
        
        super().__init__(config)

        self.runtime_settings = get_runtime_settings()
        self.num_knowledge_agents = num_knowledge_agents
        self.num_subject_agents = num_subject_agents
        self.social_learning_enabled = self.runtime_settings["social_learning_enabled"]

        # Fixed number of data points for consistency
        self.num_snapshots = int(self.runtime_settings.get("num_snapshots", 60))
        self.snapshot_interval_seconds = float(self.runtime_settings.get("snapshot_interval_seconds", 2.0))
        self.snapshots_recorded = 0

        # Subject visibility / information teleportation settings
        teleport_settings = self.runtime_settings.get("information_teleportation", {})
        self.teleportation_enabled = teleport_settings.get("enabled", False)
        self.teleportation_mode = teleport_settings.get("mode")
        self.active_ratio = teleport_settings.get("active_ratio", 0.2)
        self.mean_swap_time = teleport_settings.get("mean_swap_time", 10.0)
        self.next_swap_time = None
        self._appeared_subjects: set[int] = set()

        ground_truth_bundle = self.runtime_settings["ground_truth"]
        self.ground_truth_key = ground_truth_bundle.get("name")
        self.ground_truth_text = ground_truth_bundle.get("text", "")
        self.ground_truth_facts = ground_truth_bundle.get("facts", [])
        self.ground_truth_summary = ground_truth_bundle.get("summary", "")
 
        self.start_time = pg.time.get_ticks()
        self.tick_count = 0
        self._experiment_saved = False
        self.run_dir = self._make_run_dir()

    def _make_run_dir(self) -> Path:
        base_dir = Path("experiments")
        active_experiment = self.runtime_settings.get("active_experiment", "unspecified")
        learning_mode = self.runtime_settings.get("learning_mode", "self")

        teleport = self.runtime_settings.get("information_teleportation", {})
        if teleport.get("enabled"):
            active_ratio = teleport.get("active_ratio", 0.2)
            mean_swap_time = teleport.get("mean_swap_time", 10.0)
            run_dirname = f"{active_experiment}__{learning_mode}__ratio{active_ratio}__swap{int(mean_swap_time)}s"
        else:
            run_dirname = f"{active_experiment}__{learning_mode}"

        profile_dir = base_dir / run_dirname
        profile_dir.mkdir(parents=True, exist_ok=True)

        existing_runs = [p for p in profile_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
        next_idx = 1
        if existing_runs:
            try:
                next_idx = max(int(p.name.split("_")[-1]) for p in existing_runs) + 1
            except ValueError:
                next_idx = 1

        run_dir = profile_dir / f"run_{next_idx:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _HeadlessSimulation__update_positions(self):
        env_w = self.runtime_settings["environment"]["width"]
        env_h = self.runtime_settings["environment"]["height"]
        for sprite in self._agents.sprites():
            agent: Agent = sprite

            linear_speed, angular_velocity = agent.get_velocities()
            agent.actuator.update_position(linear_speed, angular_velocity)
            # Bounce off walls: reflect direction and nudge back inside so agents don't get stuck
            self._bounce_agent_off_walls(agent, env_w, env_h)

    def _bounce_agent_off_walls(self, agent: Agent, env_w: float, env_h: float):
        """If agent is outside or on the boundary, reflect its heading and clamp position inside."""
        bounced_x = False
        bounced_y = False
        if agent.pos.x < 0:
            agent.current_angle = (180 - agent.current_angle) % 360
            agent.pos.x = 0
            bounced_x = True
        elif agent.pos.x > env_w:
            agent.current_angle = (180 - agent.current_angle) % 360
            agent.pos.x = env_w
            bounced_x = True
        if agent.pos.y < 0:
            agent.current_angle = (360 - agent.current_angle) % 360
            agent.pos.y = 0
            bounced_y = True
        elif agent.pos.y > env_h:
            agent.current_angle = (360 - agent.current_angle) % 360
            agent.pos.y = env_h
            bounced_y = True

    def clean_llm_result(self, result: str) -> str:
        """Clean and validate LLM result. Returns empty string if invalid."""
        if not result:
            return ""
        
        cleaned = result.strip()
        while cleaned and cleaned[0] in '"\'\\ ' and cleaned[-1] in '"\'\\ ':
            cleaned = cleaned.strip().strip('"\'\\')
        empty_responses = {'""', "''", '\"\"', "\'\'", "null", "None", "N/A", "n/a", "empty"}
        if cleaned.lower() in empty_responses or cleaned in empty_responses:
            return ""
        
        # Skip very short nonsense
        if len(cleaned) < 5:
            return ""
        
        # Must contain at least one alphanumeric character
        if not any(c.isalnum() for c in cleaned):
            return ""
        
        return cleaned
    def run(self):
        """
        Run the simulation until all snapshots are recorded.
        Duration = num_snapshots × snapshot_interval_seconds
        """
        self._running = True
        self.next_snapshot_time = self.snapshot_interval_seconds
        
        total_duration = self.num_snapshots * self.snapshot_interval_seconds
        logger.info("Starting experiment: %d snapshots x %.1fs = %.1fs total", self.num_snapshots, self.snapshot_interval_seconds, total_duration)
        
        while self._running:
            self.tick()
            self.tick_count += 1

            elapsed_seconds = self._elapsed_sim_seconds()
            
            if self.teleportation_enabled:
                if self.teleportation_mode == "dynamic_pool_with_reemission":
                    self.dynamic_pool_with_reemission_update()
                elif self.teleportation_mode == "dynamic_pool_without_reemission":
                    self.dynamic_pool_without_reemission_update()
            
            # Check if it's time for a snapshot
            if elapsed_seconds >= self.next_snapshot_time and self.snapshots_recorded < self.num_snapshots:
                self.record_snapshot()
                self.next_snapshot_time += self.snapshot_interval_seconds
            
            # Stop when we have all snapshots
            if self.snapshots_recorded >= self.num_snapshots:
                logger.info("All %d snapshots recorded. Ending experiment.", self.num_snapshots)
                self.stop()
                break

        self.save_experiment_data()

        return self._metrics


    def record_snapshot(self):
        """Record a snapshot for ALL agents. Always records exactly one data point per agent."""
        for agent in self._agents:
            if agent.role == "KNOWLEDGE_AGENT":
                summary = " ".join(agent.t_summary)
                cleaned_summary = self.clean_llm_result(summary)
                agent.summary_history.append(cleaned_summary if cleaned_summary else "")

        self.snapshots_recorded += 1
        logger.info("Snapshot %d/%d recorded", self.snapshots_recorded, self.num_snapshots)

    def _elapsed_sim_seconds(self) -> float:
        fps = getattr(self.config, "fps", None) if hasattr(self, "config") else None
        if fps:
            try:
                fps_value = float(fps)
                if fps_value > 0:
                    return self.tick_count / fps_value
            except (TypeError, ValueError):
                pass
        return (pg.time.get_ticks() - self.start_time) / 1000.0

    def initialize_dynamic_pool_with_reemission(self):
        """
        Initialize exponential swap pool mode.
        All subjects are already spawned. Activate round(active_ratio * N) of them
        at random positions, hide the rest, then sample the first swap time from
        an exponential distribution with mean = mean_swap_time.
        No per-subject lifetimes are used — a single global timer drives all swaps.
        """
        subjects = [a for a in self._agents if getattr(a, "role", None) == "SUBJECT"]
        total = len(subjects)
        target_active = max(1, round(self.active_ratio * total))

        random.shuffle(subjects)
        active_subjects = subjects[:target_active]
        inactive_subjects = subjects[target_active:]

        env_width = self.runtime_settings["environment"]["width"]
        env_height = self.runtime_settings["environment"]["height"]

        for subject in active_subjects:
            x = random.uniform(50, env_width - 50)
            y = random.uniform(50, env_height - 50)
            subject.pos.update((x, y))
            subject.set_visible(True)

        for subject in inactive_subjects:
            subject.set_visible(False)

        current_time = self._elapsed_sim_seconds()
        self.next_swap_time = current_time + random.expovariate(1.0 / self.mean_swap_time)

        logger.info("Dynamic pool (with reemission) initialized: %d/%d subjects active (%.0f%% ratio), first swap at t=%.2fs",
                    target_active, total, self.active_ratio * 100, self.next_swap_time)

    def dynamic_pool_with_reemission_update(self):
        """
        Single-event Poisson swap: do nothing until next_swap_time is reached, then
        hide exactly one random active subject and show exactly one random inactive subject.
        Immediately sample the next swap time from Exp(mean_swap_time).
        """
        if self.next_swap_time is None:
            return

        current_time = self._elapsed_sim_seconds()
        if current_time < self.next_swap_time:
            return

        subjects = [a for a in self._agents if getattr(a, "role", None) == "SUBJECT"]
        active = [s for s in subjects if getattr(s, "visible", True)]
        inactive = [s for s in subjects if not getattr(s, "visible", True)]

        if active and inactive:
            outgoing = random.choice(active)
            outgoing.set_visible(False)

            incoming = random.choice(inactive)
            env_width = self.runtime_settings["environment"]["width"]
            env_height = self.runtime_settings["environment"]["height"]
            incoming.pos.update((random.uniform(50, env_width - 50), random.uniform(50, env_height - 50)))
            incoming.set_visible(True)

            new_active_count = len(active)  # net change is zero: −1 +1
            logger.debug("Dynamic pool swap (reemission): 1 out, 1 in. Active: %d/%d", new_active_count, len(subjects))

        self.next_swap_time = current_time + random.expovariate(1.0 / self.mean_swap_time)

    def initialize_dynamic_pool_without_reemission(self):
        """
        Initialize exponential one-time pool mode.
        Identical to initialize_dynamic_pool_with_reemission, but the initially-active subjects
        are recorded in _appeared_subjects so they can never re-enter once swapped out.
        """
        subjects = [a for a in self._agents if getattr(a, "role", None) == "SUBJECT"]
        total = len(subjects)
        target_active = max(1, round(self.active_ratio * total))

        random.shuffle(subjects)
        active_subjects = subjects[:target_active]
        inactive_subjects = subjects[target_active:]

        env_width = self.runtime_settings["environment"]["width"]
        env_height = self.runtime_settings["environment"]["height"]

        for subject in active_subjects:
            x = random.uniform(50, env_width - 50)
            y = random.uniform(50, env_height - 50)
            subject.pos.update((x, y))
            subject.set_visible(True)
            self._appeared_subjects.add(id(subject))

        for subject in inactive_subjects:
            subject.set_visible(False)

        current_time = self._elapsed_sim_seconds()
        self.next_swap_time = current_time + random.expovariate(1.0 / self.mean_swap_time)

        never_appeared_count = len(inactive_subjects)
        logger.info("Dynamic pool (without reemission) initialized: %d/%d subjects active (%.0f%% ratio), %d never-appeared in reserve, first swap at t=%.2fs",
                    target_active, total, self.active_ratio * 100, never_appeared_count, self.next_swap_time)

    def dynamic_pool_without_reemission_update(self):
        """
        Single-event Poisson swap with permanent retirement.
        When the timer fires: one random active subject is hidden and permanently retired
        (added to _appeared_subjects so it will never be swapped back in).
        A replacement is drawn only from subjects that have NEVER appeared before.
        If no never-appeared subjects remain, the active count decreases by one.
        """
        if self.next_swap_time is None:
            return

        current_time = self._elapsed_sim_seconds()
        if current_time < self.next_swap_time:
            return

        subjects = [a for a in self._agents if getattr(a, "role", None) == "SUBJECT"]
        active = [s for s in subjects if getattr(s, "visible", True)]
        never_appeared = [s for s in subjects if not getattr(s, "visible", True) and id(s) not in self._appeared_subjects]

        if active:
            outgoing = random.choice(active)
            outgoing.set_visible(False)
            # outgoing was already in _appeared_subjects from init or a previous swap-in

            if never_appeared:
                incoming = random.choice(never_appeared)
                env_width = self.runtime_settings["environment"]["width"]
                env_height = self.runtime_settings["environment"]["height"]
                incoming.pos.update((random.uniform(50, env_width - 50), random.uniform(50, env_height - 50)))
                incoming.set_visible(True)
                self._appeared_subjects.add(id(incoming))
                logger.debug("One-time swap: 1 retired, 1 new in. Active: %d/%d, never-appeared remaining: %d",
                             len(active), len(subjects), len(never_appeared) - 1)
            else:
                logger.debug("One-time swap: 1 retired, pool exhausted. Active: %d/%d", len(active) - 1, len(subjects))

        self.next_swap_time = current_time + random.expovariate(1.0 / self.mean_swap_time)

    def save_experiment_data(self):
        """Save experiment data and plot to the experiments directory"""
        if self._experiment_saved:
            logger.warning("Experiment data already saved; skipping duplicate save.")
            return
        try:
            # Generate logical timestamps: [2, 4, 6, ..., 120] based on snapshot interval
            interval = self.snapshot_interval_seconds
            timestamps = [int((i + 1) * interval) for i in range(self.num_snapshots)]
            
            data = {
                "timestamps": timestamps,
                "snapshot_interval_seconds": interval,
                "num_snapshots": self.num_snapshots,
                "agents": {}
            }
            
            for agent in self._agents:
                if getattr(agent, "role", None) == "KNOWLEDGE_AGENT":
                    agent_key = str(agent.id)
                    summaries = list(getattr(agent, "summary_history", []))
                    logger.debug("Agent %s: %d summaries", agent_key, len(summaries))
                    data["agents"][agent_key] = {
                        "summaries": {str(t): summaries[i] if i < len(summaries) else "" for i, t in enumerate(timestamps)},
                        "llm_context": agent.llm.get_context_stats(),
                    }

            run_dir = self.run_dir
            active_experiment = self.runtime_settings.get("active_experiment", "unspecified")
            learning_mode = self.runtime_settings.get("learning_mode", "self")
            swarm_type = self.runtime_settings.get("swarm_type") or ("social_learning" if self.social_learning_enabled else "self_learning")

            metadata = {
                "created_at_utc": datetime.utcnow().isoformat() + "Z",
                "active_experiment": active_experiment,
                "learning_mode": learning_mode,
                "swarm_type": swarm_type,
                "num_knowledge_agents": self.num_knowledge_agents,
                "num_subject_agents": self.num_subject_agents,
                "social_learning_enabled": self.social_learning_enabled,
                "num_snapshots": self.num_snapshots,
                "ground_truth_key": self.ground_truth_key,
                "ground_truth_summary": self.ground_truth_summary,
                "ground_truth_snippet_count": len(self.runtime_settings["ground_truth"].get("snippets", [])),
                "context": self.runtime_settings.get("context", {}),
                "information_teleportation": {
                    "enabled": self.teleportation_enabled,
                    "mode": self.teleportation_mode,
                    "active_ratio": self.active_ratio,
                    "mean_swap_time": self.mean_swap_time,
                } if self.teleportation_enabled else {"enabled": False},
                "movement": self.runtime_settings.get("movement", {}),
            }
 
            output = {**metadata, **data}
            run_path = run_dir / "run.json"
            with open(run_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            logger.info("Saved experiment to %s", run_dir)
            self._experiment_saved = True
        except Exception as e:
            logger.error("Failed to save experiment data: %s", e)

