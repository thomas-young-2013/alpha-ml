import time
import pickle
import numpy as np
import os
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from litesmac.scenario.scenario import Scenario
from litesmac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer
from alphaml.utils.constants import MAX_INT
from tqdm import tqdm


class MONO_MAB_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)

        self.B = kwargs['runtime'] if ('runtime' in kwargs and kwargs['runtime'] is not None and
                                       kwargs['runtime'] > 0) else None
        if self.B is not None:
            self.iter_num = MAX_INT
        else:
            self.iter_num = MAX_INT if ('runcount' not in kwargs or kwargs['runcount'] is None) else kwargs['runcount']

        self.estimator_arms = list(self.config_space.keys())
        self.mode = kwargs['update_mode'] if 'update_mode' in kwargs else 2

        self.C = 10 if 'param' not in kwargs else kwargs['param']
        self.task_name = kwargs['task_name'] if 'task_name' in kwargs else 'default'
        self.result_file = self.task_name + '_mm_bandit_%d_smac.data' % self.mode

        self.smac_containers = dict()
        self.cnts = dict()
        self.rewards = dict()
        self.updated_rewards = dict()
        self.configs_list = list()
        self.config_values = list()
        # Runtime estimate for each arm.
        self.runtime_est = dict()
        self.bar = tqdm(range(self.iter_num),
                        bar_format='{desc} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        # self.bar = tqdm(range(self.iter_num))
        for estimator in self.estimator_arms:
            # Scenario object
            config_space = self.config_space[estimator]
            estimator_hp = CategoricalHyperparameter("estimator", [estimator], default_value=estimator)
            config_space.add_hyperparameter(estimator_hp)
            scenario_dict = {
                'abort_on_first_run_crash': False,
                "run_obj": "quality",
                "cs": config_space,
                "deterministic": "true"
            }

            smac = SMAC(scenario=Scenario(scenario_dict),
                        rng=np.random.RandomState(self.seed), tae_runner=self.evaluator)
            self.smac_containers[estimator] = smac
            self.cnts[estimator] = 0
            self.rewards[estimator] = list()
            self.updated_rewards[estimator] = list()
            self.runtime_est[estimator] = 0.

    def run(self):

        self.logger.info('Start task: %s' % self.task_name)

        arm_set = list(self.estimator_arms)
        T = self.iter_num
        iter_num = 0
        tmp_iter = 0
        duration = self.C

        while True:
            # Pull each arm exactly once.
            tmp_iter += 1
            p, q = list(), list()
            es_flag = False

            for arm in arm_set:
                self.logger.info('Choosing to optimize %s arm' % arm)
                iter_start_time = time.time()
                self.smac_containers[arm].iterate()
                self.runtime_est[arm] += (time.time() - iter_start_time)
                runhistory = self.smac_containers[arm].solver.runhistory

                # Observe the reward.
                runkeys = list(runhistory.data.keys())
                for key in runkeys[self.cnts[arm]:]:
                    reward = 1 - runhistory.data[key][0]
                    self.rewards[arm].append(reward)
                    self.updated_rewards[arm].append(max(self.rewards[arm]))
                    self.configs_list.append(runhistory.ids_config[key[0]])
                    self.config_values.append(reward)

                # Determine whether to stop early.
                if len(arm_set) == 1 and len(runkeys[self.cnts[arm]:]) == 0:
                    es_flag = True

                # Record the time cost.
                time_point = time.time() - self.start_time
                tmp_list = list()
                tmp_list.append(time_point)
                for key in reversed(runkeys[self.cnts[arm] + 1:]):
                    time_point -= runhistory.data[key][1]
                    tmp_list.append(time_point)
                self.timing_list.extend(reversed(tmp_list))

                iter_run = len(runkeys) - self.cnts[arm]
                prev_num = iter_num
                iter_num += iter_run

                # Update progress bar
                if iter_num >= self.iter_num:
                    if prev_num >= self.iter_num:
                        update_iter = 0
                    else:
                        update_iter = self.iter_num - 1 - prev_num
                else:
                    update_iter = iter_run
                self.bar.set_description("%s evaluated" % arm)
                self.bar.update(update_iter)

                self.cnts[arm] = len(runhistory.data.keys())

                if self.mode == 4:
                    eval_cost = self.runtime_est[arm] / self.cnts[arm]
                    eval_cnt_left = (self.start_time + self.B - time.time()) / eval_cost
                    eval_cnt_left //= 2
                    eval_cnt_left = max(1, eval_cnt_left)
                    self.logger.info('%s: Look Forward %d Steps' % (arm.upper(), eval_cnt_left))

                acc_reward = self.updated_rewards[arm]
                if self.cnts[arm] > 2:
                    if len(acc_reward) >= duration:
                        estimated_slope = (acc_reward[-1] - acc_reward[-duration]) / duration
                    else:
                        # estimated_slope = (acc_reward[-1] - acc_reward[0]) / len(acc_reward)
                        estimated_slope = 1.

                    if self.mode == 1:
                        F = sum(acc_reward)
                        pred = sum([min(1., acc_reward[-1] + estimated_slope * (t - tmp_iter))
                                    for t in range(tmp_iter + 1, T)])
                        p.append(F + pred)
                        q.append(F + acc_reward[-1] * (T - tmp_iter))
                    elif self.mode == 2:
                        p.append(min(1., acc_reward[-1] + estimated_slope * (T - tmp_iter)))
                        q.append(acc_reward[-1])
                    elif self.mode == 3:
                        p.append(min(1., acc_reward[-1] + estimated_slope * (T - len(self.config_values))))
                        q.append(acc_reward[-1])
                    elif self.mode == 4:
                        p.append(min(1., acc_reward[-1] + estimated_slope * eval_cnt_left))
                        q.append(acc_reward[-1])
                    else:
                        raise ValueError('Invalid mode: %d.' % self.mode)
                else:
                    p.append(acc_reward[-1])
                    q.append(acc_reward[-1])
            self.logger.info('PQ estimate: %s' % dict(zip(arm_set, [[qt, pt] for qt, pt in zip(q, p)])))
            self.logger.info('Iteration %d, the best reward found is %f' % (iter_num, abs(max(self.config_values))))

            # Remove some arm.
            N = len(arm_set)
            flags = [False] * N
            for i in range(N):
                for j in range(N):
                    if i != j:
                        if p[i] < q[j]:
                            flags[i] = True

            self.logger.info('>>>>> Remove Models: %s' % [item for index, item in enumerate(arm_set) if flags[index]])
            arm_set = [item for index, item in enumerate(arm_set) if not flags[index]]

            if iter_num >= self.iter_num or es_flag:
                self.bar.update(1)
                self.bar.close()
                break

            # Check the budget.
            if self.B is not None and (time.time() - self.start_time >= self.B):
                break

        # Print the parameters in Thompson sampling.
        self.logger.info('ARM counts: %s' % self.cnts)
        self.logger.info('ARM rewards: %s' % self.rewards)

        # Print the tuning result.
        self.logger.info('MONO_BAI smbo ==> the size of evaluations: %d' % len(self.configs_list))
        if len(self.configs_list) > 0:
            id = np.argmax(self.config_values)
            self.logger.info('MONO_BAI smbo ==> The time points: %s' % self.timing_list)
            self.logger.info('MONO_BAI smbo ==> The best performance found: %f' % abs(self.config_values[id]))
            self.logger.info('MONO_BAI smbo ==> The best HP found: %s' % self.configs_list[id])
            self.incumbent = self.configs_list[id]

            # Save the experimental results.
            data = dict()
            data['ts_cnts'] = self.cnts
            data['ts_rewards'] = self.rewards
            data['configs'] = self.configs_list
            data['perfs'] = self.config_values
            data['time_cost'] = self.timing_list
            dataset_id = self.result_file.split('_')[0]
            save_dir = 'data/%s/' % dataset_id
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open('data/%s/' % dataset_id + self.result_file, 'wb') as f:
                pickle.dump(data, f)
