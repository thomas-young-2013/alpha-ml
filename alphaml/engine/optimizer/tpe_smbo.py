import time
import pickle
import os
import datetime
from datetime import timezone
import numpy as np
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer


class TPE_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)
        self.task_name = kwargs['task_name'] if 'task_name' in kwargs else 'default'
        self.result_file = self.task_name + '_hyperopt.data'
        self.estimators = list(self.config_space.keys())
        self.config_space = {
            'estimator': hp.choice('estimator',
                                   [(estimator, self.config_space[estimator]) for estimator in self.estimators])}
        self.trials = Trials()
        self.runcount = int(1e10) if 'runcount' not in kwargs or kwargs['runcount'] is None else kwargs['runcount']

        def objective(x):
            return {
                'loss': self.evaluator(x),
                'status': STATUS_OK,
                'config': x
            }

        self.objective = objective
        self.configs_list = []
        self.config_values = []

    def run(self):
        self.logger.info('Start task: %s' % self.task_name)

        fmin(self.objective, self.config_space, tpe.suggest, self.runcount, trials=self.trials)

        for trial in self.trials.trials:
            config = trial['result']['config']
            perf = - trial['result']['loss']
            time_taken = trial['book_time'].replace(tzinfo=timezone.utc).astimezone(
                tz=None).timestamp() - self.start_time
            self.configs_list.append(config)
            self.config_values.append(perf)
            self.timing_list.append(time_taken)

        self.logger.info('TPE ==> the size of evaluations: %d' % len(self.configs_list))
        if len(self.configs_list) > 0:
            id = np.argmax(self.config_values)
            self.incumbent = self.configs_list[id]

            self.logger.info('TPE ==> The time points: %s' % self.timing_list)
            self.logger.info('TPE ==> The best performance found: %f' % abs(max(self.config_values)))
            self.logger.info('TPE ==> The best HP found: %s' % self.incumbent)

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
