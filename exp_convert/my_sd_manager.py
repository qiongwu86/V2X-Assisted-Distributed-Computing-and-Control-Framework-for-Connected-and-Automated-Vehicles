from metadrive.manager.scenario_data_manager import ScenarioDataManager
import numpy as np


class MySDManager(ScenarioDataManager):

    def _get_scenario(self, i):
        ret = super()._get_scenario(i)

        # temp = dict()
        # temp['321'] = ret['tracks']['321']
        # temp['256'] = ret['tracks']['256']
        # temp['256']['state']['valid'] = np.ones_like(temp['256']['state']['valid'])
        #
        # ret['tracks'] = temp

        # temp = dict()
        # temp['98'] = ret['map_features']['98']
        # ret['map_features'] = temp

        return ret
