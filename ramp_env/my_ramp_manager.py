from metadrive.component.pgblock.ramp import InRampOnStraight
import math
import numpy as np
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.pgblock.create_pg_block_utils import ExtendStraightLane, CreateRoadFrom, CreateAdverseRoad, \
    create_bend_straight
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


class MARampPGMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MARampMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


class MARampMap(PGMap):
    def _generate(self):
        exit_length = self.config["exit_length"]

        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        # Build a first-block
        last_block = FirstPGBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config[self.LANE_NUM],
            parent_node_path,
            physics_world,
            length=exit_length,
            remove_negative_lanes=True
        )
        self.blocks.append(last_block)

        last_block = MyInRampOnStraight(
            1,
            last_block.get_socket(index=0),
            self.road_network,
            random_seed=1,
            ignore_intersection_checking=False,
            remove_negative_lanes=True,
            # **extra_kwargs
        )

        last_block.construct_block(parent_node_path, physics_world)
        self.blocks.append(last_block)


class MyInRampOnStraight(InRampOnStraight):
    ID = "r"
    EXIT_LEN = 20
    RAMP_LEN = 100
    CONNECT_TARGET_LEN = 10
    ACC_LEN = 40
    OVER_LEN = 20
    ANGLE = 10
    RADIUS = 1
    SPEED_LIMIT = 1000

    def _try_plug_into_previous_block(self) -> bool:
        no_cross = True

        # extend road and acc raod part, part 0
        self.set_part_idx(0)
        sin_angle = math.sin(np.deg2rad(self.ANGLE))
        cos_angle = math.cos(np.deg2rad(self.ANGLE))
        tan_angle = math.tan(np.deg2rad(self.ANGLE))
        longitude_len = self.CONNECT_TARGET_LEN + self.RAMP_LEN - self.EXIT_LEN

        extend_lane = ExtendStraightLane(
            self.positive_basic_lane, longitude_len, [PGLineType.BROKEN, PGLineType.CONTINUOUS]
        )
        extend_road = Road(self.pre_block_socket.positive_road.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            extend_lane,
            self.positive_lane_num,
            extend_road,
            self.block_network,
            self._global_network,
            side_lane_line_type=PGLineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        extend_road.get_lanes(self.block_network)[-1].line_types = [
            PGLineType.BROKEN if self.positive_lane_num != 1 else PGLineType.CONTINUOUS, PGLineType.CONTINUOUS
        ]
        # no_cross = CreateAdverseRoad(
        #     extend_road,
        #     self.block_network,
        #     self._global_network,
        #     ignore_intersection_checking=self.ignore_intersection_checking
        # ) and no_cross
        # _extend_road = -extend_road
        # left_lane_line = PGLineType.NONE if self.positive_lane_num == 1 else PGLineType.BROKEN
        # _extend_road.get_lanes(self.block_network)[-1].line_types = [left_lane_line, PGLineType.SIDE]

        # main acc part
        acc_side_lane = ExtendStraightLane(
            extend_lane, self.ACC_LEN + self.lane_width, [extend_lane.line_types[0], PGLineType.SIDE]
        )
        acc_road = Road(extend_road.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            acc_side_lane,
            self.positive_lane_num,
            acc_road,
            self.block_network,
            self._global_network,
            side_lane_line_type=PGLineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        # no_cross = CreateAdverseRoad(
        #     acc_road,
        #     self.block_network,
        #     self._global_network,
        #     ignore_intersection_checking=self.ignore_intersection_checking
        # ) and no_cross
        left_line_type = PGLineType.CONTINUOUS if self.positive_lane_num == 1 else PGLineType.BROKEN
        acc_road.get_lanes(self.block_network)[-1].line_types = [left_line_type, PGLineType.BROKEN]

        # socket part
        socket_side_lane = ExtendStraightLane(acc_side_lane, self.OVER_LEN, acc_side_lane.line_types)
        socket_road = Road(acc_road.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            socket_side_lane,
            self.positive_lane_num,
            socket_road,
            self.block_network,
            self._global_network,
            side_lane_line_type=PGLineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        # no_cross = CreateAdverseRoad(
        #     socket_road,
        #     self.block_network,
        #     self._global_network,
        #     ignore_intersection_checking=self.ignore_intersection_checking
        # ) and no_cross
        self.add_sockets(PGBlock.create_socket_from_positive_road(socket_road))

        # ramp part, part 1
        self.set_part_idx(1)
        lateral_dist = tan_angle * self.CONNECT_TARGET_LEN
        end_point = extend_lane.position(-self.EXIT_LEN + self.RAMP_LEN, lateral_dist + self.lane_width)
        start_point = extend_lane.position(-self.EXIT_LEN, lateral_dist + self.lane_width)
        straight_part = StraightLane(
            start_point, end_point, self.lane_width, self.LANE_TYPE, speed_limit=self.SPEED_LIMIT
        )
        straight_road = Road(self.add_road_node(), self.add_road_node())
        self.block_network.add_lane(straight_road.start_node, straight_road.end_node, straight_part)
        no_cross = (
            not check_lane_on_road(
                self._global_network,
                straight_part,
                0.95,
                ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross
        self.add_respawn_roads(straight_road)

        # p1 road 0, 1
        bend_1, connect_part = create_bend_straight(
            straight_part,
            (self.CONNECT_TARGET_LEN * tan_angle - 2 * self.RADIUS * (1 - cos_angle)) / sin_angle,
            self.RADIUS,
            np.deg2rad(self.ANGLE),
            False,
            self.lane_width,
            self.LANE_TYPE,
            speed_limit=self.SPEED_LIMIT
        )
        bend_1_road = Road(straight_road.end_node, self.add_road_node())
        connect_road = Road(bend_1_road.end_node, self.add_road_node())
        self.block_network.add_lane(bend_1_road.start_node, bend_1_road.end_node, bend_1)
        self.block_network.add_lane(connect_road.start_node, connect_road.end_node, connect_part)
        no_cross = (
            not check_lane_on_road(
                self._global_network, bend_1, 0.95, ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross
        no_cross = (
            not check_lane_on_road(
                self._global_network,
                connect_part,
                0.95,
                ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross

        # p1, road 2, 3
        bend_2, acc_lane = create_bend_straight(
            connect_part,
            self.ACC_LEN,
            self.RADIUS,
            np.deg2rad(self.ANGLE),
            True,
            self.lane_width,
            self.LANE_TYPE,
            speed_limit=self.SPEED_LIMIT
        )
        acc_lane.line_types = [PGLineType.BROKEN, PGLineType.CONTINUOUS]
        bend_2_road = Road(connect_road.end_node, self.road_node(0, 0))  # end at part1 road 0, extend road
        self.block_network.add_lane(bend_2_road.start_node, bend_2_road.end_node, bend_2)
        self.block_network.add_lane(acc_road.start_node, acc_road.end_node, acc_lane)
        no_cross = (
            not check_lane_on_road(
                self._global_network, bend_2, 0.95, ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross
        no_cross = (
            not check_lane_on_road(
                self._global_network, acc_lane, 0.95, ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross

        # p1, road 4, small circular to decorate
        merge_lane, _ = create_bend_straight(
            acc_lane, 10, self.lane_width / 2, np.pi / 2, False, self.lane_width,
            (PGLineType.BROKEN, PGLineType.CONTINUOUS)
        )
        self.block_network.add_lane(Decoration.start, Decoration.end, merge_lane)

        return no_cross


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
            (0, 0),
            # (0, -5),
            (100, -5 - 10 * np.tan(np.deg2rad(10))),
            (110, -5),
            (110, 0),
            (150, 0),
            (150, -5)
        ]
        for p in pos3:
            self.spawn_object(TrafficCone, position=p, heading_theta=0)
        # for longitude in range(0, 150, 10):
        #     self.spawn_object(TrafficCone, position=[longitude, 7], heading_theta=0)


class MARampSpawnManager(SpawnManager):
    file_name = '../output_dir/solve_info/on_ramp'
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

    spawn_merge = dict(
        identifier='1r1_0_|1r1_1_|0|0',
        config=dict(
            spawn_lane_index=('1r1_0_', '1r1_1_', 0),
            spawn_longitude=0.0,
            spawn_lateral=0
        )
    )

    @classmethod
    def _get_veh_info(cls, v_id):
        v_info = cls.init_state_info[v_id]
        return v_info['new_state'], v_info['control']

    def _auto_fill_spawn_roads_randomly(self, spawn_roads):
        agent_configs = []

        for v_id in self.veh_id_set:
            state, _ = self._get_veh_info(v_id)
            x, y, phi, v = state
            temp = copy.deepcopy(self.spawn_main) if y > -1.0 else copy.deepcopy(self.spawn_merge)
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

