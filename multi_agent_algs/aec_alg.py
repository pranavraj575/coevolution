from pettingzoo import AECEnv
from multi_agent_alg import MultiAgentAlgorithm
from ppo.PPO import WorkerPPO
from stable_baselines3.ppo import MlpPolicy


class AECAlgorithm(MultiAgentAlgorithm):
    def __init__(self,
                 env: AECEnv,
                 workers,
                 policy=MlpPolicy,
                 DefaultWorkerClass=WorkerPPO,
                 worker_infos=None,
                 **worker_kwargs,
                 ):
        """
        initializes multi agent algorithm with specified workers
        if any agent_ids are unspecified, uses DefaultWorkerClass to initalize them
        Args:
            parallel_env: Pettingzoo AECEnv to use
            workers: dict of agentid -> worker
                trainable workers must inherit
                    multi_agent_algs.off_policy:OffPolicyAlgorithm
                    or multi_agent_algs.on_policy:OnPolicyAlgorithm
                untrainable workers must have a get_action (obs -> action) method
            worker_infos: dict of agentid -> (worker info dict)
                worker info dict contains
                    DICT_TRAIN: bool (whether or not to train worker)

            policy: Type of policy to use for stableBaselines algorithm
            DefaultWorkerClass: class to use to initialize workers
            **worker_kwargs: kwargs to use to initializw workers
        """
        super().__init__(
            policy=policy,
            env=env,
            DefaultWorkerClass=DefaultWorkerClass,
            workers=workers,
            worker_infos=worker_infos,
            **worker_kwargs,
        )
        self.agent_records = dict()
        self.trainable_workers = set(self.get_trainable_workers())

    def learn_episode(self,
                      total_timesteps,
                      number_of_eps=None,
                      strict_timesteps=True,
                      callbacks=None,
                      ):
        """
        learn episode, collects total_timesteps steps then trains
        Args:
            total_timesteps: number of timesteps to collect
            number_of_eps: if specified, overrides total_timesteps, and instead collects this number of episodes
            strict_timesteps: if true, breaks an episode in the middle if timesteps are over
            callbacks:
        Returns: number of collected timesteps
        """
        if callbacks is None:
            callbacks = {agent: None for agent in self.workers}

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
        steps_so_far = 0
        continue_rollout = True
        if self.reset_env:
            episodes_completed = -1
            # if we are resetting immedieately, we should start at -1
        else:
            episodes_completed = 0

        while continue_rollout:
            if self.reset_env:
                # if we have just completed an episode, increase the episode counter
                # we want to terminate the loop before resetting env if possible
                episodes_completed += 1
            if number_of_eps is not None:
                # counter for number of episodes to do
                if episodes_completed >= number_of_eps:
                    continue_rollout = False
            if strict_timesteps and steps_so_far >= total_timesteps:
                continue_rollout = False
            if not continue_rollout:
                break

            continue_rollout = False
            if self.reset_env:
                self.env.reset()
                self.reset_env = False
                # reset records
                self.agent_records = dict()

            # inner loop, should run until env ends
            for agent in self.env.agent_iter():
                # let previous timestep be t and current timestep be t+1
                # obs,reward,info from between timesteps t and t+1
                # termination/truncation means there is no future step for this agent
                new_obs, reward, termination, truncation, info = self.env.last()

                term = termination or truncation
                if term:
                    self.reset_env = True

                if agent in self.agent_records:
                    # if we have a previous timestep, finish the rollout, and remove it from memory
                    ((old_obs, old_action),
                     (old_rollout_1_info, old_rollout_2_info)) = self.agent_records.pop(agent)

                    # rollout 3 (end of loop)
                    rollout_3_info = self.workers[agent].rollout_3(
                        action=old_action,
                        new_obs=new_obs,
                        reward=reward,
                        termination=termination,
                        truncation=truncation,
                        info=info,
                        init_learn_info=local_init_learn_info[agent],
                        init_rollout_info=local_init_rollout_info[agent],
                        rollout_1_info=old_rollout_1_info,
                        rollout_2_info=old_rollout_2_info,
                    )

                    steps_so_far += 1
                    if strict_timesteps and steps_so_far >= total_timesteps:
                        # if we go over, we should break immedeatley
                        continue_rollout = False
                        break

                if agent in self.trainable_workers:
                    # rollout 1 (start of loop)
                    rollout_1_info = self.workers[agent].rollout_1(
                        init_learn_info=local_init_learn_info[agent],
                        init_rollout_info=local_init_rollout_info[agent],
                    )

                    # rollout 2 (middle of loop, action selection)
                    act, rollout_2_info = self.workers[agent].rollout_2(
                        obs=new_obs,
                        init_learn_info=local_init_learn_info[agent],
                        init_rollout_info=local_init_rollout_info[agent],
                        rollout_1_info=rollout_1_info,
                    )
                    self.env.step(act)
                    self.agent_records[agent] = (
                        (new_obs, act), (rollout_1_info, rollout_2_info)
                    )
                else:
                    act = self.workers[agent].get_action(obs=new_obs)
                    self.env.step(act)

        # end rollout
        local_end_rollout_info = dict()
        for agent in self.get_trainable_workers():
            end_rollout_info = self.workers[agent].end_rollout(
                init_learn_info=local_init_learn_info[agent],
                init_rollout_info=local_init_rollout_info[agent],
            )
            local_end_rollout_info[agent] = end_rollout_info

        for agent in self.get_trainable_workers():
            self.workers[agent].finish_learn(
                init_learn_info=local_init_learn_info[agent],
                end_rollout_info=local_end_rollout_info[agent],
            )

        return steps_so_far, max(0, episodes_completed)
