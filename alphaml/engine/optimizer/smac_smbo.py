import time
import pickle
import os
import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer
from alphaml.engine.components.components_manager import ComponentsManager


class SMAC_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)
        self.task_name = kwargs['task_name'] if 'task_name' in kwargs else 'default'
        self.result_file = self.task_name + '_smac.data'

        # Scenario object
        config_space = ComponentsManager.build_hierarchical_configspace(self.config_space)
        scenario_dict = {
            'abort_on_first_run_crash': False,
            "run_obj": "quality",
            "cs": config_space,
            "deterministic": "true"
        }
        self.runtime = None
        if 'runtime' in kwargs and kwargs['runtime'] is not None and kwargs['runtime'] > 0:
            scenario_dict['wallclock_limit'] = kwargs['runtime']
            self.runtime = kwargs['runtime']
        else:
            if 'runcount' in kwargs and kwargs['runcount'] is not None and kwargs['runcount'] > 0:
                scenario_dict['runcount-limit'] = kwargs['runcount']
            else:
                raise ValueError('Limit value error!')

        self.scenario = Scenario(scenario_dict)
        self.smac = SMAC(scenario=self.scenario, rng=np.random.RandomState(self.seed), tae_runner=self.evaluator)
        self.configs_list = list()
        self.config_values = list()

    def run(self):
        self.logger.info('Start task: %s' % self.task_name)
        self.smac.optimize()
        runhistory = self.smac.solver.runhistory
        trajectory = self.smac.solver.intensifier.traj_logger.trajectory
        self.incumbent = self.smac.solver.incumbent

        # Fetch the results.
        runkeys = list(runhistory.data.keys())
        for key in runkeys:
            reward = 1 - runhistory.data[key][0]
            self.configs_list.append(runhistory.ids_config[key[0]])
            self.config_values.append(reward)

        # Record the time cost.
        time_point = time.time() - self.start_time
        tmp_list = list()
        tmp_list.append(time_point)
        for key in reversed(runkeys[1:]):
            time_point -= runhistory.data[key][1]
            tmp_list.append(time_point)
        self.timing_list.extend(reversed(tmp_list))

        self.logger.info('SMAC smbo ==> the size of evaluations: %d' % len(self.configs_list))
        if len(self.configs_list) > 0:
            self.logger.info('SMAC smbo ==> The time points: %s' % self.timing_list)
            self.logger.info('SMAC smbo ==> The best performance found: %f' % max(self.config_values))
            self.logger.info('SMAC smbo ==> The best HP found: %s' % self.incumbent)

            # Save the experimental results.
            data = dict()
            data['configs'] = self.configs_list
            data['perfs'] = self.config_values
            data['time_cost'] = self.timing_list
            dataset_id = self.result_file.split('_')[0]
            save_dir = 'data/%s/' % dataset_id
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open('data/%s/' % dataset_id + self.result_file, 'wb') as f:
                pickle.dump(data, f)
