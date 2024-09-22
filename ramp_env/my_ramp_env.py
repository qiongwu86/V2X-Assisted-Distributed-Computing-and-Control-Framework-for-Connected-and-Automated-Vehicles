from my_ramp_manager import MARampPGMapManager, MARampSpawnManager, ConeExampleManager
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.ramp import InRampOnStraight
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.utils import Config
from typing import Union
import numpy as np
from metadrive.policy.idm_policy import IDMPolicy
import time
from my_ramp_policy import MyRampPolicy

MARampConfig = dict(
    physics_world_step_size=1e-1,
    decision_repeat=1,
    spawn_roads=[
        Road(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2),
        # Road(FirstPGBlock.NODE_3, InRampOnStraight.node(1, 0, 0)),
        Road(InRampOnStraight.node(1, 1, 0), InRampOnStraight.node(1, 1, 1))
    ],
    # destination=InRampOnStraight.node(1, 0, 2),
    map_config=dict(exit_length=20, lane_num=1),
    top_down_camera_initial_x=120,
    top_down_camera_initial_y=-50,
    top_down_camera_initial_z=10
)


class MultiAgentRampEnv(MultiAgentMetaDrive):

    def __init__(self, config: Union[dict, None] = None):
        super().__init__(config)

    def set_camera_state(self, pos=None, look=None):
        self.main_camera.camera_x = pos[0]
        self.main_camera.camera_x = pos[1]
        self.main_camera.top_down_camera_height = pos[2]
        self.main_camera.camera.setPos(*pos) if pos is not None else None
        # self.main_camera.camera_hpr = look if look is not None else None
        self.main_camera.camera.lookAt(*look) if look is not None else None

    def switch_to_top_down_view(self):
        self.main_camera.stop_track(bird_view_on_current_position=False)

    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MARampConfig, allow_add_new_key=True)

    def setup_engine(self):
        super(MultiAgentMetaDrive, self).setup_engine()
        self.engine.register_manager("spawn_manager", MARampSpawnManager())
        self.engine.update_manager("map_manager", MARampPGMapManager())
        # self.engine.update_manager("traffic_manager", ConeExampleManager())


if __name__ == "__main__":
    env = MultiAgentRampEnv(
        {
            "agent_policy": MyRampPolicy,
            # "agent_policy": IDMPolicy,
            "horizon": 10000,
            "num_agents": 10,
            "use_render": True,
            "debug": False,
            "allow_respawn": False,
            "manual_control": False,
            "map_config": dict(
                exit_length=20,
                lane_num=1,
                lane_width=5.0
            ),
            "out_of_road_done": False,
            "crash_done": False
        }
    )
    stop = False
    obs, _ = env.reset()
    env.switch_to_top_down_view()
    env.set_camera_state(pos=(160, 5, 30), look=(110, 0, 0))
    time.sleep(1)
    time_step = 0
    while not stop:
        time_step += 1
        env.set_camera_state(
            pos=(180, -10, 10), look=(110, 0, 0)
        )
        acts = {agent: np.zeros(2) for agent in obs.keys()}
        # print(acts)
        obs, reward, terminated, truncated, info = env.step(acts)
        # print(sum([r for r in reward.values()]))
        stop = terminated["__all__"]
        # print(stop)
        fig = env.render(
            mode='topdown',
                   window=False,
                   camera_position=env.current_map.get_center_point(),
                   screen_record=True,
                   screen_size=(700, 100))
    env.top_down_renderer.generate_gif()
    env.close()
