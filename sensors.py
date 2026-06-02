import random
from runtime_config import get_runtime_settings

_SETTINGS = get_runtime_settings()

class Sensor:

    def __init__(self, agent):
        self.agent = agent

    def exchange_context_with_agents(self, new_agent_objects):
        # Only exchange with the first neighbor to avoid redundant merges in the same tick
        first_agent = new_agent_objects[0]
        first_agent_summary = first_agent.t_summary[-1] if len(first_agent.t_summary) > 0 else ""
        current_agent_summary = self.agent.t_summary[-1] if len(self.agent.t_summary) > 0 else ""

        if first_agent_summary:
            self.agent.t_received.append(first_agent_summary)
            self.agent.summarize_interaction()
        if current_agent_summary:
            first_agent.t_received.append(current_agent_summary)


    def collect_information_from_subjects(self, new_agent_objects):
        for agent in new_agent_objects:
            if agent.role == "SUBJECT":
                self.agent.p.append(agent.info)

class Actuator:

    def __init__(self, agent):
        self.agent = agent
        self.agent.current_angle = random.uniform(0, 360)

    def update_position(self, linear_speed: int, angular_velocity: int):
        self.agent.current_angle += angular_velocity
        self.agent.current_angle %= 360

        self.agent.move.from_polar((linear_speed, self.agent.current_angle))  # polar → cartesian
        self.agent.pos += self.agent.move
        
        