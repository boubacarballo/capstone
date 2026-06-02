from vi import Agent
from sensors import Actuator, Sensor
from runtime_config import get_runtime_settings

_SETTINGS = get_runtime_settings()
_MOVEMENT_CFG = _SETTINGS.get("movement", {}) or {}
class Subject:
    def __init__(self, information: str, position: tuple):
        self.information = information
        self.position = position

    def get_information(self):
        return self.information

    def get_position(self):
        return self.position
    
    def get_subject(self):
        return self.subject

class SubjectAgent(Agent):
    def __init__(self, *args, info: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.role = "SUBJECT"
        self.info = info
        self.actuator = Actuator(self)
        self.sensor = Sensor(self)  # needed so moving subjects bounce off walls
        self.visible = True  # when False, knowledge agents skip this subject
        self._movement_enabled = bool(_MOVEMENT_CFG.get("enabled", False))
        self._movement_speed = float(_MOVEMENT_CFG.get("speed", 1.5))
        self._movement_angular_velocity = float(_MOVEMENT_CFG.get("angular_velocity", 5.0))

    def update(self):
        # Static subject: no behavior
        pass
    
    def set_visible(self, visible: bool):
        self.visible = visible
        if hasattr(self, 'image') and self.image is not None:
            self.image.set_alpha(255 if visible else 40)  # dim sprite when invisible

    def get_velocities(self):
        if not self._movement_enabled:
            return 0, 0

        # Wall bounce is handled in Environment._bounce_agent_off_walls
        linear_speed = self._movement_speed
        angular_velocity = 0.0
        return linear_speed, angular_velocity


