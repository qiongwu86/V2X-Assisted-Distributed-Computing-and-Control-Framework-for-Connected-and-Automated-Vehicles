from my_intersection_manager import MyIntersectionPGMapManager, MyIntersectionSpawnManager, ConeExampleManager
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.utils import Config
from typing import Union
import numpy as np
import time
from my_intersection_policy import MyIntersectionPolicy

MARampConfig = dict(
    physics_world_step_size=1e-1,
    decision_repeat=1,
    spawn_roads=[
        # Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
        -Road(InterSection.node(1, 0, 0), InterSection.node(1, 0, 1)),
        -Road(InterSection.node(1, 1, 0), InterSection.node(1, 1, 1)),
        -Road(InterSection.node(1, 2, 0), InterSection.node(1, 2, 1)),
    ],
    num_agents=30,
    map_config=dict(exit_length=60, lane_num=1, lane_width=5.0),
    top_down_camera_initial_x=40.25,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=50,
    out_of_road_done=False
)


class MultiAgentRampEnv(MultiAgentMetaDrive):

    def __init__(self, config: Union[dict, None] = None):
        super().__init__(config)

    def set_camera_state(self, pos=None, look=None):
        self.main_camera.camera.setPos(*pos) if pos is not None else None
        self.main_camera.camera.lookAt(*look) if look is not None else None

    def switch_to_top_down_view(self):
        self.main_camera.stop_track(bird_view_on_current_position=False)

    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MARampConfig, allow_add_new_key=True)

    def setup_engine(self):
        super(MultiAgentMetaDrive, self).setup_engine()
        # self.engine.register_manager("spawn_manager", SpawnManager())
        self.engine.register_manager("spawn_manager", MyIntersectionSpawnManager())
        self.engine.update_manager("map_manager", MyIntersectionPGMapManager())
        # self.engine.update_manager("traffic_manager", ConeExampleManager())

    # def done_function(self, vehicle_id):


if __name__ == "__main__":
    env = MultiAgentRampEnv(
        {
            "agent_policy": MyIntersectionPolicy,
            # "agent_policy": IDMPolicy,
            "horizon": 10000,
            "num_agents": 12,
            "use_render": True,
            "debug": False,
            "allow_respawn": False,
            "manual_control": False,
            "map_config": dict(
                exit_length=36-4.25/2,
                lane_num=1,
                lane_width=4.25
            ),
        }
    )
    stop = False
    obs, _ = env.reset()
    env.switch_to_top_down_view()
    d = np.sqrt(40.25**2+ (4.25/2)**2)
    env.set_camera_state(pos=(40.25, 4.25/2, 20), look=(40.25, 4.25/2, 0))
    time.sleep(1)
    time_step = 0
    while not stop:
        time_step += 1
        # env.set_camera_state(pos=(130, 0, 50), look=(130, 0, 0))
        env.set_camera_state(
            pos=(40.25+d*np.cos(np.deg2rad(135)-time_step/100), 4.25/2-d*np.sin(np.deg2rad(135)-time_step/100), 30),
            look=(40.25, 4.25/2, 0)
        )
        acts = {agent: np.zeros(2) for agent in obs.keys()}
        # print(acts)
        obs, reward, terminated, truncated, info = env.step(acts)
        # print(sum([r for r in reward.values()]))
        stop = terminated["__all__"] or time_step > 100
        # print(stop)
        env.render(
            mode='topdown',
                   window=False,
                   camera_position=env.current_map.get_center_point(),
                   screen_record=True,
                   screen_size=(500, 500))
    env.top_down_renderer.generate_gif()
    env.close()
