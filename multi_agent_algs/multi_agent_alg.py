from pettingzoo import AECEnv, ParallelEnv
from common import DumEnv
from utils.dict_keys import DICT_TRAIN
from typing import Union
from ppo.PPO import WorkerPPO
from stable_baselines3.ppo import MlpPolicy


class MultiAgentAlgorithm:
    def __init__(self,
                 env: Union[AECEnv, ParallelEnv],
                 workers,
                 policy=MlpPolicy,
                 DefaultWorkerClass=WorkerPPO,
                 worker_infos=None,
                 **worker_kwargs
                 ):
        """
        initializes multi agent algorithm with specified workers
        if any agent_ids are unspecified, uses DefaultWorkerClass to initalize them
        Args:
            env: Pettingzoo env to use
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
        if workers is None:
            workers = dict()
        if worker_infos is None:
            worker_infos = dict()
        for agent in env.agents:
            if agent not in worker_infos:
                worker_infos[agent] = {
                    DICT_TRAIN: True
                }

            dumenv = DumEnv(action_space=env.action_space(agent=agent),
                            obs_space=env.observation_space(agent=agent),
                            )
            if agent not in workers:
                workers[agent] = DefaultWorkerClass(policy=policy,
                                                    env=dumenv,
                                                    **worker_kwargs,
                                                    )
            elif worker_infos[agent].get(DICT_TRAIN, True):
                # in this case, we should probably set the environment anyway
                workers[agent].set_env(dumenv)

        self.workers = workers
        self.worker_info = worker_infos
        self.env = env
        self.reset_env = True  # should reset env next time

    def learn(self,
              total_timesteps,
              number_of_eps=None,
              number_of_eps_per_learning_step=1,
              strict_timesteps=True,
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
            strict_timesteps: if true, breaks an episode in the middle if timesteps are over
            callbacks:
        Returns:
        """
        # always start learning with resetting environment
        self.reset_env = True
        timestep_counter = 0
        episode_counter = 0
        while True:
            local_num_eps = None
            if number_of_eps is not None:
                local_num_eps = min(number_of_eps_per_learning_step, number_of_eps)
            timesteps, episodes = self.learn_episode(total_timesteps=total_timesteps,
                                                     number_of_eps=local_num_eps,
                                                     strict_timesteps=strict_timesteps,
                                                     callbacks=callbacks,
                                                     )
            timestep_counter += timesteps
            episode_counter += episodes
            if number_of_eps is not None:
                # if this is specified, train for this number of eps
                if episode_counter >= number_of_eps:
                    break
            else:
                # otherwise, break if we run out of timesteps
                if timestep_counter >= total_timesteps:
                    break
        return timestep_counter, episode_counter

    def _get_worker_iter(self, trainable):
        """
        Args:
            trainable: if true, returns trainable workers
                else, untrainable workers
        Returns: iterable of trainable or untrainable workers
        """
        for agent in self.workers:
            is_trainable = self.worker_info[agent].get(DICT_TRAIN, True)
            if is_trainable == trainable:  # either both true or both false
                yield agent

    def get_trainable_workers(self):
        return self._get_worker_iter(trainable=True)

    def get_untrainable_workers(self):
        return self._get_worker_iter(trainable=False)

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
        Returns: number of collected timesteps, number of collected episodes
        """
        raise NotImplementedError
