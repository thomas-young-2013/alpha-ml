import time
import pickle
import numpy as np
import os
from datetime import timezone
from hyperopt import hp, tpe, base, FMinIter, Trials, STATUS_OK
from hyperopt.fmin import generate_trials_to_calculate
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer
from alphaml.utils.constants import MAX_INT


def get_iter(fn, space, algo, max_evals, trials=None, rstate=None,
             pass_expr_memo_ctrl=None,
             catch_eval_exceptions=False,
             verbose=0,
             points_to_evaluate=None,
             max_queue_len=1,
             show_progressbar=False,
             ):
    if rstate is None:
        env_rseed = os.environ.get('HYPEROPT_FMIN_SEED', '')
        if env_rseed:
            rstate = np.random.RandomState(int(env_rseed))
        else:
            rstate = np.random.RandomState()

    if trials is None:
        if points_to_evaluate is None:
            trials = base.Trials()
        else:
            assert type(points_to_evaluate) == list
            trials = generate_trials_to_calculate(points_to_evaluate)

    domain = base.Domain(fn, space,
                         pass_expr_memo_ctrl=pass_expr_memo_ctrl)

    rval = FMinIter(algo, domain, trials, max_evals=max_evals,
                    rstate=rstate,
                    verbose=verbose,
                    max_queue_len=max_queue_len,
                    show_progressbar=show_progressbar)
    rval.catch_eval_exceptions = catch_eval_exceptions
    return rval


class MONO_MAB_TPE_SMBO(BaseOptimizer):
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
        self.result_file = self.task_name + '_mm_bandit_%d_tpe.data' % self.mode

        self.tpe_containers = dict()
        self.cnts = dict()
        self.rewards = dict()
        self.updated_rewards = dict()
        self.configs_list = list()
        self.config_values = list()
        # Runtime estimate for each arm.
        self.runtime_est = dict()

        def objective(x):
            return {
                'loss': self.evaluator(x),
                'status': STATUS_OK,
                'config': x
            }

        self.objective = objective
        for estimator in self.estimator_arms:
            # Scenario object
            config_space = self.config_space[estimator]
            config_space = {
                'estimator': hp.choice('estimator',
                                       [(estimator, config_space)])}
            trials = Trials()
            fmin_iter = get_iter(self.objective, config_space, tpe.suggest, MAX_INT, trials=trials)
            self.tpe_containers[estimator] = fmin_iter
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
                # iterate
                next(self.tpe_containers[arm])
                self.runtime_est[arm] += (time.time() - iter_start_time)
                trials = self.tpe_containers[arm].trials.trials

                # Observe the reward.
                for trial in trials[self.cnts[arm]:]:
                    reward = 1 - trial['result']['loss']
                    self.rewards[arm].append(reward)
                    self.updated_rewards[arm].append(max(self.rewards[arm]))
                    self.configs_list.append(trial['result']['config'])
                    self.config_values.append(reward)

                # Determine whether to stop early.
                if len(arm_set) == 1 and len(trials[self.cnts[arm]:]) == 0:
                    es_flag = True

                # Record the time cost.
                for trial in trials[self.cnts[arm]:]:
                    time_taken = trial['book_time'].replace(tzinfo=timezone.utc).astimezone(
                        tz=None).timestamp() - self.start_time
                    self.timing_list.append(time_taken)

                iter_num += (len(trials) - self.cnts[arm])
                self.cnts[arm] = len(trials)

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
            self.logger.info('Iteration %d, the best reward found is %f' % (iter_num, max(self.config_values)))

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
            self.logger.info('MONO_BAI smbo ==> The best performance found: %f' % self.config_values[id])
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
