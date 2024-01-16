import matplotlib.artist
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Text, Tuple
from dynamic_models import KinematicModel


class ColorSet:
    color_set: List = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple']
    color_ptr: int = 0

    @staticmethod
    def get_next_color() -> Text:
        ColorSet.color_ptr += 1
        if ColorSet.color_ptr == len(ColorSet.color_set):
            ColorSet.color_ptr = 0
        return ColorSet.color_set[ColorSet.color_ptr]


class MapInfo:
    def __init__(self, _solid: List[Dict], _dashed: List[Dict] = None):
        self._solid = _solid
        self._dashed = _dashed if _dashed is not None else list()

    def plot_map(self, _ax, _lw=0.25):
        self.plot_solid(_ax, _lw)
        for _one_dashed in self._dashed:
            _ax.plot(_one_dashed['x'], _one_dashed['y'], linestyle='--', linewidth=_lw, color='black')

    def plot_solid(self, _ax, _lw=0.25):
        for _one_solid in self._solid:
            _ax.plot(_one_solid['x'], _one_solid['y'], linestyle='-', linewidth=_lw, color='black')


def _rand_in_range(low: float, high: float) -> float:
    assert high > low
    return np.random.random() * (high - low) + low


def star_traj(
        _veh_num: int = 3,
        _speed: float = 3.0,
        _init_road_length: float = 30.0,
        _over_road_length: float = 30.0,
        _random: bool = True,
) -> Tuple[List[np.ndarray], int]:
    _speed_100ms = _speed / 10
    inter_rad = 2 * np.pi / _veh_num

    rand_low = -0.5
    rand_high = 0.5

    _max_step = 99999

    def _gen_one_traj(_rad) -> np.ndarray:
        x = np.arange(
            -_init_road_length + _rand_in_range(rand_low, rand_high),
            _over_road_length + _rand_in_range(rand_low, rand_high),
            _speed_100ms
        )
        y = np.zeros_like(x)
        phi = np.zeros_like(x)
        v = np.ones_like(x) * _speed
        origin_traj = np.vstack((x, y, phi, v)).transpose()
        # ratate
        one_traj = np.zeros_like(origin_traj)
        one_traj[:, 0] = np.cos(_rad) * origin_traj[:, 0]
        one_traj[:, 1] = -np.sin(_rad) * origin_traj[:, 0]
        one_traj[:, 2] = -_rad * np.ones_like(origin_traj[:, 2])
        one_traj[:, 3] = origin_traj[:, 3]
        return one_traj

    ret = list()
    for i in range(_veh_num):
        _one_traj = _gen_one_traj(i * inter_rad)
        _max_step = min(_max_step, _one_traj.shape[0])
        ret.append(_one_traj)

    return ret, _max_step


def speed_change(ref_dist: float, ref_speed: float, obj_dist: float) -> float:
    return (ref_speed * obj_dist) / ref_dist


def cross_traj(
        _speed: float = 5.0,
        _init_road_length: float = 20.0,
        _over_road_length: float = 50.0,
        _random: bool = True,
        _road_width: float = 5.0,
        _round: int = 2,
        _round_distance: float = 15
) -> Tuple[List[np.ndarray], int, np.ndarray]:
    _speed_100ms = _speed / 10
    _half_road_width = 0.5 * _road_width

    '''
        |2|
    ____| | ____
    3          1
    ____    ____
        | |
        |0|
    '''

    def _ft(_f: int, _t: int):
        t = (4 - _f) % 4
        return (_f + t) % 4, (_t + t) % 4

    def _gen_one_traj(_from: int, _to: int, _init_length_round: float, _over_length_round: float):
        assert _from != _to and _from in (0, 1, 2, 3) and _to in (0, 1, 2, 3), "{}, {}".format(_from, _to)
        init_from, init_to = _ft(_from, _to)
        assert init_from == 0
        # part 1
        y = np.arange(-_half_road_width - _init_length_round, -_half_road_width, _speed_100ms)
        x = np.zeros_like(y)
        phi = np.ones_like(y) * np.pi * 0.5
        v = _speed * np.ones_like(y)
        traj_part1 = np.vstack((x, y, phi, v)).transpose()
        # part 2
        if init_to == 2:
            # traj_part2_1
            y = np.arange(-_half_road_width, _half_road_width,
                          speed_change(0.25 * 2 * np.pi * _half_road_width, _speed_100ms, _road_width))
            x = np.zeros_like(y)
            phi = np.ones_like(y) * np.pi * 0.5
            v = _speed * np.ones_like(y)
            traj_part2_1 = np.vstack((x, y, phi, v)).transpose()
            # traj_part2_2
            y = np.arange(_half_road_width, _half_road_width + _over_length_round, _speed_100ms)
            x = np.zeros_like(y)
            phi = np.ones_like(y) * np.pi * 0.5
            v = _speed * np.ones_like(y)
            traj_part2_2 = np.vstack((x, y, phi, v)).transpose()
            traj_part2 = np.vstack((traj_part2_1, traj_part2_2))
        else:
            # traj_part2_1
            _rad_speed_100ms = _speed_100ms / _half_road_width
            x = _half_road_width - np.cos(np.arange(0, 0.5 * np.pi, _rad_speed_100ms)) * _half_road_width
            y = -_half_road_width + np.sin(np.arange(0, 0.5 * np.pi, _rad_speed_100ms)) * _half_road_width
            phi = 0.5 * np.pi - np.arange(0, 0.5 * np.pi, _rad_speed_100ms)
            v = _speed * np.ones_like(x)
            traj_part2_1 = np.vstack((x, y, phi, v)).transpose()
            # traj_part2_2
            x = np.arange(_half_road_width, _half_road_width + _over_length_round, _speed_100ms)
            y = np.zeros_like(x)
            phi = np.zeros_like(x)
            v = np.ones_like(x) * _speed
            traj_part2_2 = np.vstack((x, y, phi, v)).transpose()

            if init_to != 1:
                traj_part2_1[:, 0] = -traj_part2_1[:, 0]
                traj_part2_1[:, 2] = np.pi - traj_part2_1[:, 2]
                traj_part2_2[:, 0] = -traj_part2_2[:, 0]
                traj_part2_2[:, 2] = np.pi * np.ones_like(traj_part2_2[:, 2])
            traj_part2 = np.vstack((traj_part2_1, traj_part2_2))

        _traj = np.vstack((traj_part1, traj_part2))
        for _ in range(_from):
            _traj = _rotate_90(_traj)

        return _traj

    def _rotate_90(one_traj: np.ndarray) -> np.ndarray:
        new_one_traj = np.zeros_like(one_traj)
        new_one_traj[:, 0] = -one_traj[:, 1]
        new_one_traj[:, 1] = one_traj[:, 0]
        new_one_traj[:, 2] = one_traj[:, 2] + np.pi * 0.5
        new_one_traj[:, 3] = one_traj[:, 3]
        return new_one_traj

    def _gen_one_round(_init_length_round: float, _over_length_round: float) -> Tuple[List, int, np.ndarray]:
        round_max_step = 99999
        one_round_all_traj = []
        from_set = np.array([0, 1, 2, 3])
        to_set = np.array([0, 1, 2, 3])
        _rest_shuffle_times = 100
        while any(from_set == to_set) and _rest_shuffle_times:
            np.random.shuffle(to_set)
            _rest_shuffle_times -= 1

        for _f, _t in zip(from_set, to_set):
            _one_traj = _gen_one_traj(_f, _t, _init_length_round, _over_length_round)
            round_max_step = min(round_max_step, _one_traj.shape[0])
            one_round_all_traj.append(_one_traj)
        round_info = np.vstack((from_set, to_set))
        return one_round_all_traj, round_max_step, round_info

    all_traj = []
    info_rounds = np.zeros((_round, 2, 4))
    max_step = 99999
    for i in range(_round):
        init_length = _init_road_length + i * _round_distance
        over_length = _over_road_length - i * _round_distance
        round_traj, _max_step, one_round_info = _gen_one_round(init_length, over_length)
        max_step = min(max_step, _max_step)
        all_traj = all_traj + round_traj
        info_rounds[i] = one_round_info

    return all_traj, max_step, info_rounds


def cross_traj_double_lane(
        _speed: float = 5.0,
        _init_road_length: float = 20.0,
        _over_road_length: float = 70.0,
        _random: bool = True,
        _road_width: float = 8.0,
        _round: int = 2,
        _round_distance: float = 15,
        _log_file: Text = None
) -> Tuple[List[np.ndarray], int, np.ndarray, MapInfo]:
    """
        |2|
    ____| | ____
    3          1
    ____    ____
        | |
        |0|
    """

    def _ft(_f: int, _t: int):
        t = (4 - _f) % 4
        return (_f + t) % 4, (_t + t) % 4

    def _gen_one_traj(_from: int, _to: int, _init_length_round: float, _over_length_round: float):
        assert _from != _to and _from in (0, 1, 2, 3) and _to in (0, 1, 2, 3), "{}, {}".format(_from, _to)
        init_from, init_to = _ft(_from, _to)
        assert init_from == 0
        _half_road_width = 0.5 * _road_width
        _quarter_road_width = 0.25 * _road_width
        _to_axis = (np.sqrt(2) - 1) * _half_road_width
        _r_small = (2 - np.sqrt(2)) * _half_road_width
        _r_big = np.sqrt(2) * _half_road_width
        _r_small_speed = _speed
        _straight_speed = speed_change(0.25 * 2 * _r_small, _r_small_speed, _road_width)
        _r_big_speed = speed_change(0.25 * 2 * _r_small, _r_small_speed, 0.25 * 2 * _r_big)

        # part 1
        y = np.arange(-_half_road_width - _init_length_round, -_half_road_width, _speed / 10)
        x = np.ones_like(y) * _to_axis
        phi = np.ones_like(y) * np.pi * 0.5
        v = _speed * np.ones_like(y)
        traj_part1 = np.vstack((x, y, phi, v)).transpose()
        # part 2
        if init_to == 1:
            # traj_part2_1
            _rad_speed_100ms = (_r_small_speed / 10) / _r_small
            x = _half_road_width - np.cos(np.arange(0, 0.5 * np.pi, _rad_speed_100ms)) * _r_small
            y = -_half_road_width + np.sin(np.arange(0, 0.5 * np.pi, _rad_speed_100ms)) * _r_small
            phi = 0.5 * np.pi - np.arange(0, 0.5 * np.pi, _rad_speed_100ms)
            v = _r_small_speed * np.ones_like(x)
            traj_part2_1 = np.vstack((x, y, phi, v)).transpose()
            # traj_part2_2
            x = np.arange(_half_road_width, _half_road_width + _over_length_round, _speed / 10)
            y = -np.ones_like(x) * _to_axis
            phi = np.zeros_like(x)
            v = np.ones_like(x) * _speed
            traj_part2_2 = np.vstack((x, y, phi, v)).transpose()
        elif init_to == 2:
            # traj_part2_1
            y = np.arange(-_half_road_width, _half_road_width, _straight_speed / 10)
            x = np.ones_like(y) * _to_axis
            phi = np.ones_like(x) * 0.5 * np.pi
            v = _straight_speed * np.ones_like(x)
            traj_part2_1 = np.vstack((x, y, phi, v)).transpose()
            # traj_part2_2
            y = np.arange(_half_road_width, _half_road_width + _over_length_round, _speed / 10)
            x = np.ones_like(y) * _to_axis
            phi = np.ones_like(x) * 0.5 * np.pi
            v = np.ones_like(x) * _speed
            traj_part2_2 = np.vstack((x, y, phi, v)).transpose()
        elif init_to == 3:
            # traj_part2_1
            _rad_speed_100ms = (_r_big_speed / 10) / _r_big
            x = -_half_road_width + np.cos(np.arange(0, 0.5 * np.pi, _rad_speed_100ms)) * _r_big
            y = -_half_road_width + np.sin(np.arange(0, 0.5 * np.pi, _rad_speed_100ms)) * _r_big
            phi = 0.5 * np.pi + np.arange(0, 0.5 * np.pi, _rad_speed_100ms)
            v = _r_big_speed * np.ones_like(x)
            traj_part2_1 = np.vstack((x, y, phi, v)).transpose()
            # traj_part2_2
            x = np.arange(-_half_road_width, -_half_road_width - _over_length_round, -_speed / 10)
            y = np.ones_like(x) * _to_axis
            phi = np.ones_like(x) * np.pi
            v = np.ones_like(x) * _speed
            traj_part2_2 = np.vstack((x, y, phi, v)).transpose()
        else:
            raise ValueError

        traj_part2 = np.vstack((traj_part2_1, traj_part2_2))

        _traj = np.vstack((traj_part1, traj_part2))
        for _ in range(_from):
            _traj = _rotate_90(_traj)

        return _traj

    def _rotate_90(one_traj: np.ndarray) -> np.ndarray:
        new_one_traj = np.zeros_like(one_traj)
        new_one_traj[:, 0] = -one_traj[:, 1]
        new_one_traj[:, 1] = one_traj[:, 0]
        new_one_traj[:, 2] = one_traj[:, 2] + np.pi * 0.5
        new_one_traj[:, 3] = one_traj[:, 3]
        return new_one_traj

    def _gen_one_round(
            _init_length_round: float,
            _over_length_round: float,
            _to_set: np.ndarray = None
    ) -> Tuple[List, int, np.ndarray]:
        round_max_step = 99999
        one_round_all_traj = []
        from_set = np.array([0, 1, 2, 3])
        if _to_set is None:
            to_set = np.array([0, 1, 2, 3])
            _rest_shuffle_times = 100
            while any(from_set == to_set) and _rest_shuffle_times:
                np.random.shuffle(to_set)
                _rest_shuffle_times -= 1
        else:
            to_set = _to_set

        for _f, _t in zip(from_set, to_set):
            _one_traj = _gen_one_traj(_f, _t, _init_length_round, _over_length_round)
            round_max_step = min(round_max_step, _one_traj.shape[0])
            one_round_all_traj.append(_one_traj)
        round_info = np.vstack((from_set, to_set))
        return one_round_all_traj, round_max_step, round_info

    if _log_file is not None:
        _log_file_mat = np.load(_log_file)
        _round = _log_file_mat.shape[0]
    else:
        _log_file_mat = None

    def _gen_map_info() -> MapInfo:
        # generate one and rotate to get others
        _road_length = max(_init_road_length, _over_road_length)
        _solid: List = [
            dict(
                x=np.array([0.5 * _road_width, 0.5 * _road_width]),
                y=np.array([-0.5 * _road_width - _road_length, -0.5 * _road_width])
            ),
            dict(
                x=np.array([0.5 * _road_width, 0.5 * _road_width + _road_length]),
                y=np.array([-0.5 * _road_width, -0.5 * _road_width])
            )
        ]
        _dashed: List = [
            dict(
                x=np.array([0.5 * _road_width, 0.5 * _road_width + _road_length]),
                y=np.array([0, 0])
            ),
        ]

        for _i in range(3):
            _new_solid = dict(x=-_solid[-2]['y'], y=_solid[-2]['x'])
            _solid.append(_new_solid)
            _new_solid = dict(x=-_solid[-2]['y'], y=_solid[-2]['x'])
            _solid.append(_new_solid)
            _new_dashed = dict(x=-_dashed[-1]['y'], y=_dashed[-1]['x'])
            _dashed.append(_new_dashed)

        return MapInfo(_solid, _dashed)

    all_traj = []
    info_rounds = np.zeros((_round, 2, 4))
    max_step = 99999
    for i in range(_round):
        init_length = _init_road_length + i * _round_distance
        over_length = _over_road_length - i * _round_distance
        if _log_file is not None:
            round_traj, _max_step, one_round_info = _gen_one_round(init_length, over_length,
                                                                   _to_set=_log_file_mat[i][1])
        else:
            round_traj, _max_step, one_round_info = _gen_one_round(init_length, over_length)
        max_step = min(max_step, _max_step)
        all_traj = all_traj + round_traj
        info_rounds[i] = one_round_info

    return all_traj, max_step, info_rounds, _gen_map_info()


def multi_cross(
        _points: int = 5,
        _speed: float = 4.0,
        _init_road_length: float = 10.0,
        _over_road_length: float = 20.0,
        _x_delta: float = 7.0,
        _cross_region_length: float = 50
) -> Tuple[List[np.ndarray], int, np.ndarray, MapInfo]:
    def _gen_from_to() -> Tuple[np.ndarray, np.ndarray]:
        _from_set = np.array([i for i in range(_points)])
        _to_set = np.array([i for i in range(_points)])
        _rest_shuffle_times = 100
        while any(_from_set == _to_set) and _rest_shuffle_times:
            np.random.shuffle(_to_set)
            _rest_shuffle_times -= 1
            if _rest_shuffle_times == 0:
                raise ValueError
        return _from_set, _to_set

    def _gen_one_traj(_f: int, _t: int) -> Tuple[np.ndarray, int]:
        _speed_100ms = _speed / 10
        # part 1
        y = np.arange(0.0, _init_road_length, _speed_100ms)
        x = np.ones_like(y) * (_f * _x_delta)
        phi = np.ones_like(y) * np.pi * 0.5
        v = _speed * np.ones_like(y)

        traj_p1 = np.vstack((x, y, phi, v)).transpose()
        # part 2
        _part2_point_num = int(_cross_region_length / _speed_100ms)
        _delta_temp = (_t - _f) * _x_delta / _part2_point_num
        x = np.array([_f * _x_delta + i * _delta_temp for i in range(_part2_point_num)])
        _delta_temp = _cross_region_length / _part2_point_num
        y = np.array([_init_road_length + i * _delta_temp for i in range(_part2_point_num)])
        phi = np.zeros_like(x)
        v = np.zeros_like(x)
        traj_p2 = np.vstack((x, y, phi, v)).transpose()
        # part 3
        y = np.arange(
            _init_road_length + _cross_region_length,
            _init_road_length + _cross_region_length + _over_road_length,
            _speed_100ms
        )
        x = np.ones_like(y) * (_t * _x_delta)
        phi = np.ones_like(y) * np.pi * 0.5
        v = _speed * np.ones_like(y)
        traj_p3 = np.vstack((x, y, phi, v)).transpose()
        _traj = np.vstack((traj_p1, traj_p2, traj_p3))
        return _traj, _traj.shape[0]

    def _gen_one_traj_map(_f: int, _t: int) -> List[Dict]:
        _map_of_traj = list()
        _map_of_traj.append(dict(
            x=np.array([_f * _x_delta, _f * _x_delta]),
            y=np.array([0.0, _init_road_length])
        ))
        _map_of_traj.append(dict(
            x=np.array([_f * _x_delta, _t * _x_delta]),
            y=np.array([_init_road_length, _init_road_length + _cross_region_length])
        ))
        _map_of_traj.append(dict(
            x=np.array([_t * _x_delta, _t * _x_delta]),
            y=np.array([_init_road_length + _cross_region_length,
                        _init_road_length + _cross_region_length + _over_road_length])
        ))
        return _map_of_traj

    _map_solid = list()
    all_traj = []
    from_set, to_set = _gen_from_to()
    max_length = 99999
    for f, t in zip(from_set, to_set):
        one_traj, traj_len = _gen_one_traj(f, t)
        all_traj.append(one_traj)
        max_length = min(max_length, traj_len)
        _map_solid = _map_solid + _gen_one_traj_map(f, t)

    return all_traj, max_length, np.vstack((from_set, to_set)), MapInfo(_map_solid)


def gen_video_from_info(_all_info: List[Dict], _all_traj: List[np.ndarray], draw_nominal=True,
                        _map_info: MapInfo = None) -> None:
    def _get_nominal(_t: int, veh_id: int, _info: List[Dict]) -> np.ndarray:
        return _info[_t][veh_id]["nominal"][-1][0]  # just x_nominal, so [0]

    def _get_state(_t: int, veh_id: int, _info: List[Dict]) -> np.ndarray:
        return _info[_t][veh_id]["new_state"]

    def pos_fun(state: np.ndarray):
        assert state.shape == (4,)
        x, y, phi, _ = state
        W = KinematicModel.length
        H = KinematicModel.width
        k = 0.5 * np.sqrt(W ** 2 + H ** 2)
        beta = 0.5 * np.pi - phi - np.arctan(H / W)
        x_ = x - k * np.sin(beta)
        y_ = y - k * np.cos(beta)
        return x_, y_

    from matplotlib import animation, patches
    fig, ax = plt.subplots()
    if _map_info:
        _map_info.plot_map(ax)

    cars: Dict[int: List[matplotlib.artist.Artist]] = {
        car_id: [
            patches.Rectangle(
                (0, 0),
                KinematicModel.length,
                KinematicModel.width,
                fill=True,
                facecolor=ColorSet.get_next_color(),
                edgecolor=None
            ),
            ax.plot(
                _get_nominal(0, car_id, _all_info)[:, 0],
                _get_nominal(0, car_id, _all_info)[:, 1],
                color="red",
                linewidth=0.25
            )[0] if draw_nominal else None
        ]
        for car_id in _all_info[0].keys()
    }
    [ax.add_patch(car_obj_list[0]) for car_obj_list in cars.values()]

    # draw_range
    x_max = max([np.max(one_traj[:, 0]) for one_traj in _all_traj]) + 5
    x_min = min([np.min(one_traj[:, 0]) for one_traj in _all_traj]) - 5
    x_range_draw = (x_min, x_max)
    y_max = max([np.max(one_traj[:, 1]) for one_traj in _all_traj]) + 5
    y_min = min([np.min(one_traj[:, 1]) for one_traj in _all_traj]) - 5
    y_range_draw = (y_min, y_max)

    # ax.add_patch(ref_car)
    def update(frame):
        for car_id, car_obj_list in cars.items():
            car_rect, car_nominal = car_obj_list

            _state = _get_state(frame, car_id, _all_info)
            car_rect.set_xy(pos_fun(_state))
            car_rect.set_angle(np.rad2deg(_state[2]))

            if draw_nominal:
                _nominal = _get_nominal(frame, car_id, _all_info)
                car_nominal.set_xdata(_nominal[:, 0])
                car_nominal.set_ydata(_nominal[:, 1])

            plt.xlim(*x_range_draw)
            plt.ylim(*y_range_draw)
            ax.set_aspect('equal')
            ax.margins(0)
        return cars.values()

    anim = animation.FuncAnimation(fig, update, frames=len(_all_info), interval=100)
    writer = animation.FFMpegWriter(fps=10)
    anim.save('output_dir/video/one_veh.mp4', writer=writer)
    plt.close()


if __name__ == "__main__":
    pass