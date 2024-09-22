from metadrive.envs.scenario_env import ScenarioEnv
from my_sd_manager import MySDManager
from metadrive.manager.scenario_curriculum_manager import ScenarioCurriculumManager
from metadrive.manager.scenario_light_manager import ScenarioLightManager
from metadrive.manager.scenario_map_manager import ScenarioMapManager
from metadrive.manager.scenario_traffic_manager import ScenarioTrafficManager


class MyScenarioEnv(ScenarioEnv):

    def setup_engine(self):
        super(ScenarioEnv, self).setup_engine()
        self.engine.register_manager("data_manager", MySDManager())
        self.engine.register_manager("map_manager", ScenarioMapManager())
        if not self.config["no_traffic"]:
            self.engine.register_manager("traffic_manager", ScenarioTrafficManager())
        if not self.config["no_light"]:
            self.engine.register_manager("light_manager", ScenarioLightManager())
        self.engine.register_manager("curriculum_manager", ScenarioCurriculumManager())
