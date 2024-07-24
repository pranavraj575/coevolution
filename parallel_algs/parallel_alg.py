from pettingzoo import ParallelEnv
from parallel_algs.common import DumEnv, conform_act_shape
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, \
    is_image_space_channels_first


class ParallelAlgorithm:
    def __init__(self,
                 policy,
                 parallel_env: ParallelEnv,
                 DefaultWorkerClass,
                 workers=None,
                 worker_info=None,
                 **worker_kwargs
                 ):
        """
        Args:
            policy: Type of policy to use for stableBaselines algorithm
            parallel_env: Pettingzoo parallel env to use
            workers: dict of agentid -> worker
                trainable workers must have initialize_learn,middle_of_rollout_select,
                    get_action,end_rollout_part,finished_with_rollout
                untrainable must just have get_action
            worker_info: dict of agentid -> (worker info dict)
                worker info dict contains
                    train: bool (whether or not to treain worker)
            DefaultWorkerClass: class to use to initialize workers
            **worker_kwargs: kwargs to use to initializw workers
        """
        if workers is None:
            workers = dict()
        if worker_info is None:
            worker_info = dict()
        for agent in parallel_env.agents:
            if agent not in workers:
                dumenv = DumEnv(action_space=parallel_env.action_space(agent=agent),
                                obs_space=parallel_env.observation_space(agent=agent),
                                )
                workers[agent] = DefaultWorkerClass(policy=policy,
                                                    env=dumenv,
                                                    **worker_kwargs,
                                                    )
            if agent not in worker_info:
                worker_info[agent] = {
                    'train': True
                }
        self.workers = workers
        self.worker_info = worker_info
        self.env = parallel_env

    def learn(self,
              total_timesteps,
              number_of_eps=None,
              number_of_eps_per_learning_step=1,
              callbacks=None,
              ):
        """
        trains for total_timesteps steps
            repeatedly calls learn_episode
        Args:
            total_timesteps: number of timesteps to collect
            number_of_eps: if specified, overrides total_timesteps, and instead collects this number of episodes
            number_of_eps_per_learning_step: number of eps before each training step, default 1
                this parameter is ignored if number_of_eps is None
            callbacks:
        Returns:
        """
        while True:
            local_num_eps = None
            if number_of_eps is not None:
                local_num_eps = min(number_of_eps_per_learning_step,number_of_eps)
                number_of_eps -= number_of_eps_per_learning_step
            timesteps = self.learn_episode(total_timesteps=total_timesteps,
                                           number_of_eps=local_num_eps,
                                           callbacks=callbacks,
                                           )
            total_timesteps -= timesteps
            if number_of_eps is not None:
                # if this is specified, train for this number of eps
                if number_of_eps <= 0:
                    break
            else:
                # otherwise, break if we run out of timesteps
                if  total_timesteps <= 0:
                    break

    def _get_worker_iter(self, trainable):
        """
        Args:
            trainable: if true, returns trainable workers
                else, untrainable workers
        Returns: iterable of trainable or untrainable workers
        """
        for agent in self.workers:
            is_trainable = self.worker_info[agent].get('train', True)
            if is_trainable == trainable:  # either both true or both false
                yield agent

    def get_trainable_workers(self):
        return self._get_worker_iter(trainable=True)

    def get_untrainable_workers(self):
        return self._get_worker_iter(trainable=False)

    def learn_episode(self,
                      total_timesteps,
                      number_of_eps=None,
                      callbacks=None,
                      ):
        """
        learn episode, collects total_timesteps steps then trains
        Args:
            total_timesteps: number of timesteps to collect
            number_of_eps: if specified, overrides total_timesteps, and instead collects this number of episodes
            callbacks:
        Returns: number of collected timesteps
        """
        if callbacks is None:
            callbacks = {agent: None for agent in self.workers}
        # observations, infos = self.env.reset()

        # init learn
        local_init_learn_info = dict()
        for agent in self.get_trainable_workers():
            init_learn_info = self.workers[agent].init_learn(
                total_timesteps=total_timesteps,
                callback=callbacks[agent],
            )
            local_init_learn_info[agent] = init_learn_info

        # init rollout
        local_init_rollout_info = dict()
        for agent in self.get_trainable_workers():
            init_rollout_info = self.workers[agent].init_rollout(
                init_learn_info=local_init_learn_info[agent],
            )
            local_init_rollout_info[agent] = init_rollout_info

        continue_rollout = True
        term = True
        observations = dict()
        while continue_rollout:
            if term:
                observations, infos = self.env.reset()

            # rollout 1 (start of loop)
            local_rollout_1_info = dict()
            for agent in self.get_trainable_workers():
                rollout_1_info = self.workers[agent].rollout_1(
                    init_learn_info=local_init_learn_info[agent],
                    init_rollout_info=local_init_rollout_info[agent],
                )
                local_rollout_1_info[agent] = rollout_1_info

            # rollout 2 (middle of loop, action selection)
            actions = dict()
            local_rollout_2_info = dict()
            for agent in self.get_trainable_workers():
                act, rollout_2_info = self.workers[agent].rollout_2(
                    obs=observations[agent],
                    init_learn_info=local_init_learn_info[agent],
                    init_rollout_info=local_init_rollout_info[agent],
                    rollout_1_info=local_rollout_1_info[agent],
                )
                actions[agent] = conform_act_shape(act,
                                                   self.workers[agent].action_space,
                                                   )
                local_rollout_2_info[agent] = rollout_2_info
            # also handle the untrainable agents
            for agent in self.get_untrainable_workers():
                act = self.workers[agent].get_action(
                    obs=observations[agent]
                )
                actions[agent] = act

            observations, rewards, terminations, truncations, infos = self.env.step(actions=actions)
            truncation = any([t for (_, t) in truncations.items()])
            termination = any([t for (_, t) in terminations.items()])

            term = termination or truncation
            continue_rollout = False
            # rollout 3 (end of loop)
            for agent in self.get_trainable_workers():
                local_continue_rollout = self.workers[agent].rollout_3(
                    action=actions[agent],
                    new_obs=observations[agent],
                    reward=rewards[agent],
                    termination=termination,
                    truncation=truncation,
                    info=infos[agent],
                    init_learn_info=local_init_learn_info[agent],
                    init_rollout_info=local_init_rollout_info[agent],
                    rollout_1_info=local_rollout_1_info[agent],
                    rollout_2_info=local_rollout_2_info[agent],
                    # replay_buffer=self.workers[agent].replay_buffer,
                )
                continue_rollout = continue_rollout or local_continue_rollout

            if number_of_eps is not None:
                # counter for number of episodes to do
                number_of_eps -= 1
                if number_of_eps <= 0:
                    continue_rollout = False
        # end rollout
        local_end_rollout_info = dict()
        for agent in self.get_trainable_workers():
            end_rollout_info = self.workers[agent].end_rollout(
                init_learn_info=local_init_learn_info[agent],
                init_rollout_info=local_init_rollout_info[agent],
            )
            local_end_rollout_info[agent] = end_rollout_info

        num_collected_steps = 0
        for agent in self.get_trainable_workers():
            self.workers[agent].finish_learn(
                init_learn_info=local_init_learn_info[agent],
                end_rollout_info=local_end_rollout_info[agent],
            )
            num_collected_steps = self.workers[agent].num_collected_steps

        return num_collected_steps
