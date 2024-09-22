from metadrive.component.pgblock.ramp import InRampOnStraight
from metadrive.component.pgblock.intersection import InterSection
import math
import numpy as np
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.pgblock.create_pg_block_utils import ExtendStraightLane, CreateRoadFrom, create_bend_straight
from metadrive.component.pgblock.pg_block import PGBlock
from metadrive.component.road_network import Road
from metadrive.constants import Decoration, PGLineType
from metadrive.utils.pg.utils import check_lane_on_road
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.map.pg_map import PGMap
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.traffic_manager import TrafficManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.component.static_object.traffic_object import TrafficCone
import copy
from metadrive.utils import Config
from math import floor
import pickle


class MyIntersectionPGMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MyIntersectionMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


class MyIntersectionMap(PGMap):
    def _generate(self):
        length = self.config["exit_length"]

        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        # Build a first-block
        last_block = FirstPGBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config[self.LANE_NUM],
            parent_node_path,
            physics_world,
            length=length
        )
        self.blocks.append(last_block)

        # Build Intersection
        InterSection.EXIT_PART_LENGTH = length

        if "radius" in self.config and self.config["radius"]:
            extra_kwargs = dict(radius=self.config["radius"])
        else:
            extra_kwargs = {}
        last_block = InterSection(
            1,
            last_block.get_socket(index=0),
            self.road_network,
            random_seed=1,
            ignore_intersection_checking=False,
            **extra_kwargs
        )

        if self.config["lane_num"] > 1:
            # We disable U turn in TinyInter environment!
            last_block.enable_u_turn(True)
        else:
            last_block.enable_u_turn(False)

        last_block.construct_block(parent_node_path, physics_world)
        self.blocks.append(last_block)


class ConeExampleManager(TrafficManager):
    def reset(self):
        super(ConeExampleManager, self)
        pos = [(60.0, -8.188343312361965), (137.08800927359857, 3.5), (86.64208216692138, -4.107689879511674),
               (93.5880092735986, -3.499999999999993), (117.08800927359859, 1.75), (137.08800927359857, 1.75),
               (117.08800927359859, 7.0), (66.94592710667722, -7.5806534328502835)]
        pos2 = [
            [93.58800927, 3.5],
            [117.08800927, 3.5],
            [137.08800927, 1.75],

        ]
        pos3 = [
            (0, 0), (13, 0), (26, 0), (36, 0), (40.25, 0), (40.25, 4.25/2)
            # (27.5, 0),
            # (27.5, -2),
            # (27.5, -4),
            # (27.5, -6),
            # (30, 0),
            # (40, 0),
            # (50, 0),
            # (60, 0),
            # (70, 0),
            # (100, -5 - 10 * np.tan(np.deg2rad(10))),
            # (110, -5),
            # (110, 0),
            # (150, 0),
            # (150, -5)
        ]
        for p in pos3:
            self.spawn_object(TrafficCone, position=p, heading_theta=0)
        # for longitude in range(0, 150, 10):
        #     self.spawn_object(TrafficCone, position=[longitude, 7], heading_theta=0)


class MyIntersectionSpawnManager(SpawnManager):
    file_name = '../output_dir/solve_info/osqp_all_info_12'
    with open(file_name, 'rb') as f:
        file_obj = pickle.load(f)
        veh_id_set = set(file_obj[0])
        init_state_info = file_obj[0]

    spawn_main = dict(
        identifier='>|>>|0|0',
        config=dict(
            spawn_lane_index=('>', '>>', 0),
            spawn_longitude=0.0,
            spawn_lateral=0
        )
    )

    @classmethod
    def _get_veh_info(cls, v_id):
        v_info = cls.init_state_info[v_id]
        return v_info['new_state'] + np.array([40.25, 4.25/2, 0, 0]), v_info['control']

    def _auto_fill_spawn_roads_randomly(self, spawn_roads):
        agent_configs = []

        for v_id in self.veh_id_set:
            state, _ = self._get_veh_info(v_id)
            x, y, phi, v = state
            temp = copy.deepcopy(self.spawn_main) if y > -1.0 else copy.deepcopy(self.spawn_main)
            temp['config']['spawn_longitude'] = float(x)
            # temp['config']['spawn_velocity'] = np.array([v, 0], dtype=float)
            agent_configs.append(temp)

        safe_spawn_places = copy.deepcopy(agent_configs)

        return agent_configs, safe_spawn_places

    def reset(self):
        # set the spawn road
        ret = {}
        for idx, veh_data in enumerate(self.available_agent_configs):
            v_config = veh_data["config"]
            ret["agent{}".format(idx)] = v_config

        # set the destination/spawn point and update target_v config
        agent_configs = {}
        for agent_id, config in ret.items():
            init_config = copy.deepcopy(self._init_agent_configs[agent_id])
            if not init_config.get("_specified_spawn_lane", False):
                init_config.update(config)
            config = init_config
            if not config.get("destination", False) or config["destination"] is None:
                config = self.update_destination_for(agent_id, config)
            config['vehicle_model'] = 'ramp_veh'
            agent_configs[agent_id] = config

        self.engine.global_config["agent_configs"] = copy.deepcopy(agent_configs)

    # def get_available_respawn_places(self, map, randomize=False):
    #     return super().get_available_respawn_places(map, randomize=False)

