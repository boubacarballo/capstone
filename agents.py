import logging
from llm import LLM
from vi import Agent
from sensors import Sensor, Actuator
from concurrent.futures import Future
import random
import pygame as pg
from collections import deque
from runtime_config import get_runtime_settings

logger = logging.getLogger(__name__)

_SETTINGS = get_runtime_settings()
_ENV_WIDTH = _SETTINGS["environment"]["width"]
_ENV_HEIGHT = _SETTINGS["environment"]["height"]

class knowledgeAgent(Agent):
    def __init__(self, context_size: int = 2, social_learning_enabled: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.sensor = Sensor(self)
        self.actuator = Actuator(self) 
        self.llm = LLM()
        self.role = "KNOWLEDGE_AGENT"
        self.pos.x = random.uniform(0, _ENV_WIDTH)
        self.pos.y = random.uniform(0, _ENV_HEIGHT)
        self.surrounding_state = "ALONE"
        self.subject_state = "AWAY"
        self.p = deque(maxlen=context_size)
        self.t_summary = deque(maxlen=1)
        self.t_received = deque(maxlen=context_size)
        self.pending_future: tuple[str, Future] | None = None
        self.social_learning_enabled = social_learning_enabled
        self.summary_history = []

    def is_llm_busy(self):
        return self.pending_future is not None and not self.pending_future[1].done()

    def update(self):
        neighbors = list(self.in_proximity_performance())
        # Only count visible subjects; invisible ones are teleportation-pool placeholders
        subjects = [agent for agent in neighbors if agent.role == "SUBJECT" and getattr(agent, 'visible', True)]
        agents = [agent for agent in neighbors if agent.role == "KNOWLEDGE_AGENT"]
        number_of_neighbors = len(agents)
        number_of_subjects = len(subjects)

        if self.social_learning_enabled:
            match self.surrounding_state:
                case "ALONE":
                    if number_of_neighbors > 0:
                        if not self.is_llm_busy():
                            self.sensor.exchange_context_with_agents(agents)
                            self.summarize_interaction()
                        self.surrounding_state = "NOT_ALONE"
                        logger.debug("Agent %s is now in state NOT_ALONE", self.id)
                case "NOT_ALONE":
                    if number_of_neighbors == 0:
                        self.surrounding_state = "ALONE"

        match self.subject_state:
            case "AWAY":
                if number_of_subjects > 0:
                    logger.debug("Agent %s encountered a site", self.id)
                    if not self.is_llm_busy():
                        self.sensor.collect_information_from_subjects(subjects)
                        self.summarize_private_information()
                    self.subject_state = "NEAR_SUBJECT"
                    
            case "NEAR_SUBJECT":
                if number_of_subjects == 0:
                    self.subject_state = "AWAY"

        # summary_history is written by record_snapshot(), not here
        if self.pending_future is not None:
            task_type, future = self.pending_future
            if future.done():
                try:
                    result = future.result()
                    logger.info("Agent %s updated summary (%s): %s", self.id, task_type, result)
                    self.t_summary.append(result)
                except Exception as exc:
                    logger.error("Agent %s %s task failed: %s", self.id, task_type, exc)
                self.pending_future = None
        
        
    
    def summarize_interaction(self):
        if self.is_llm_busy():
            return
        
        own_summary = " ".join(self.t_summary) if self.t_summary else ""
        received_summaries = " ".join(self.t_received)
        
        future = self.llm.submit_interaction(summary=own_summary, received_info=received_summaries)
        if future is not None:
            self.pending_future = ("interaction", future)
    
    def summarize_private_information(self):
        if self.is_llm_busy():
            return

        own_summary = " ".join(self.t_summary) if self.t_summary else ""
        private_info = " ".join(self.p) if self.p else ""
        
        future = self.llm.submit_private_info(summary=own_summary, private_info=private_info)
        if future is not None:
            self.pending_future = ("private info", future)
    def get_velocities(self):
        # Wall bounce is handled in Environment._bounce_agent_off_walls
        linear_speed = 2
        angular_velocity = 0.0
        return linear_speed, angular_velocity


