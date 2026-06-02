from llm import LLM
from vi import Agent
from sensors import Sensor, Actuator
from concurrent.futures import Future
import random
import pygame as pg
from collections import deque
from runtime_config import get_runtime_settings

_SETTINGS = get_runtime_settings()
_ENV_WIDTH = _SETTINGS["environment"]["width"]
_ENV_HEIGHT = _SETTINGS["environment"]["height"]

class knowledgeAgent(Agent):
    def __init__(self, context_size: int = 2, social_learning_enabled: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.sensor = Sensor(self)
        self.actuator = Actuator(self) 
        self.llm = LLM(self) 
        self.role = "KNOWLEDGE_AGENT"
        self.pos.x = random.uniform(0, _ENV_WIDTH)
        self.pos.y = random.uniform(0, _ENV_HEIGHT)
        self.surrounding_state = "ALONE"
        self.object_state = "NONE"
        self.p = deque(maxlen=context_size)
        self.t_summary = deque(maxlen=1)
        self.t_received = deque(maxlen=context_size)
        self.pending_future: tuple[str, Future] | None = None
        self.social_learning_enabled = social_learning_enabled
        self.summary_history = []

    def is_llm_busy(self):
        """Check if agent is currently processing any LLM task."""
        return self.pending_future is not None and not self.pending_future[1].done()

    def update(self): # at every tick (timestep), this function will be run

        neighbors = list(self.in_proximity_performance())
        # Only interact with VISIBLE subjects (supports information teleportation)
        subjects = [agent for agent in neighbors if agent.role == "SUBJECT" and getattr(agent, 'visible', True)]
        agents = [agent for agent in neighbors if agent.role == "KNOWLEDGE_AGENT"]
        number_of_neighbors = len(agents)
        number_of_subjects = len(subjects)

        #finite state machine for surrounding state

        if self.social_learning_enabled:
            match self.surrounding_state:
                case "ALONE":
                    if number_of_neighbors > 0:
                        # here we will exchange information (if not busy)
                        if not self.is_llm_busy():
                            self.sensor.exchange_context_with_agents(agents)
                            self.summarize_interaction()
                        self.surrounding_state = "NOT_ALONE"
                        print(f"Agent {self.id} is now in the state NOT_ALONE")
                case "NOT_ALONE":
                    if number_of_neighbors == 0:
                        self.surrounding_state = "ALONE"

        #finite state machine for object state
        match self.object_state:
            case "NONE":
                if number_of_subjects > 0:
                    print("Encountered a site")
                    if not self.is_llm_busy():
                        self.sensor.collect_information_from_subjects(subjects)
                        self.summarize_private_information()
                    self.object_state = "ON_SITE"
                    
            case "ON_SITE":
                if number_of_subjects == 0:
                    self.object_state = "NONE"

        # NOTE: Only update t_summary here. summary_history is managed by record_snapshot()
        if self.pending_future is not None:
            task_type, future = self.pending_future
            if future.done():
                result = future.result()
                print(f"Agent {self.id} received {task_type} summary: {result}")
                self.t_summary.append(result)
                self.pending_future = None
        
        
    
    def summarize_interaction(self):
        """
        Summarize interaction between agents by merging this agent's summary 
        with summaries received from other agents.
        """
        if self.is_llm_busy():
            return
        
        own_summary = " ".join(self.t_summary) if self.t_summary else ""
        received_summaries = " ".join(self.t_received)
        
        print(f"Agent {self.id} merging summaries for interaction...")
        future = self.llm.submit_interaction(summary=own_summary, received_info=received_summaries)
        if future is not None:
            self.pending_future = ("interaction", future)
        print(f"Agent {self.id} scheduled interaction summarization at {pg.time.get_ticks()}ms")
    
    def summarize_private_information(self):
        """
        Summarize private information collected from sites (stored in p).
        """
        if self.is_llm_busy():
            return

        own_summary = " ".join(self.t_summary) if self.t_summary else ""
        private_info = " ".join(self.p) if self.p else ""
        
        print(f"Agent {self.id} summarizing private information...")
        future = self.llm.submit_private_info(summary=own_summary, private_info=private_info)
        if future is not None:
            self.pending_future = ("private info", future)
        print(f"Agent {self.id} scheduled private information summarization at {pg.time.get_ticks()}ms")
    def get_velocities(self):
        # Constant forward motion; wall bounce (reflect direction) is handled in Environment._bounce_agent_off_walls
        linear_speed = 2
        angular_velocity = 0.0
        return linear_speed, angular_velocity


    # removed DB logging