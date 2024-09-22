import logging
import pickle
from metadrive.policy.base_policy import BasePolicy
import numpy as np
from metadrive.scenario.parse_object_state import parse_object_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyIntersectionPolicy(BasePolicy):
    file_name = '../output_dir/solve_info/osqp_all_info_12'
    with open(file_name, 'rb') as f:
        file_obj = pickle.load(f)
        all_veh_id = set(file_obj[0].keys())
        all_time_steps = len(file_obj)

    @classmethod
    def _get_veh_info(cls, v_id, time_step):
        v_info = cls.file_obj[time_step][v_id]
        return v_info['new_state'] + np.array([40.25, 4.25/2, 0, 0]), v_info['control']

    def __init__(self, control_object, track, random_seed=None):
        super(MyIntersectionPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.v_id = self.all_veh_id.pop()
        self.start_index = 0
        self._velocity_local_frame = False

    def act(self, *args, **kwargs):
        index = max(int(self.episode_step), 0)
        if index >= self.all_time_steps:
            return None

        state, control = self._get_veh_info(self.v_id, index)
        x, y, phi, v = state
        a, psi = control

        self.control_object.set_position((x, y))
        self.control_object.set_heading_theta(phi)
        # self.control_object.set_velocity((v*np.cos(phi), v*np.sin(phi)), in_local_frame=self._velocity_local_frame)
        # self.control_object.set_steering(psi)

        return None  # Return None action so the base vehicle will not overwrite the steering & throttle
