import torch, os, shutil

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, DictRolloutBuffer

from unstable_baselines3.common import OnPolicy, OffPolicy

from src.utils.dict_keys import DICT_IS_WORKER
from src.utils.savele_baselines import (overwrite_worker, load_worker, worker_exists,
                                        overwrite_other, load_other, other_exists,
                                        overwrite_info, load_info
                                        )


class ZooCage:
    def __init__(self, zoo_dir, overwrite_zoo=True):
        self.zoo_dir = zoo_dir
        if not os.path.exists(self.zoo_dir):
            os.makedirs(self.zoo_dir)
        elif overwrite_zoo:
            self.clear()
            os.makedirs(self.zoo_dir)
        self.saved_workers = set(os.listdir(self.zoo_dir))

    def overwrite_animal(self, animal,
                         key: str,
                         info,
                         save_buffer=True,
                         save_class=True,
                         ):
        if info.get(DICT_IS_WORKER, True):
            return self.overwrite_worker(worker=animal,
                                         worker_key=key,
                                         worker_info=info,
                                         save_buffer=save_buffer,
                                         save_class=save_class,
                                         )
        else:
            return self.overwrite_other(other=animal, other_key=key, other_info=info)

    def load_animal(self, key: str, load_buffer=True):
        info = self.load_info(key=key)
        if info.get(DICT_IS_WORKER, False):
            return self.load_worker(worker_key=key,
                                    WorkerClass=None,
                                    load_buffer=load_buffer,
                                    )
        else:
            return self.load_other(other_key=key)

    ### info methods
    def overwrite_info(self,
                       key: str,
                       info,
                       ):
        overwrite_info(info=info,
                       save_path=os.path.join(self.zoo_dir, key, 'info.pkl'),
                       )

    def load_info(self,
                  key: str,
                  ):
        return load_info(save_path=os.path.join(self.zoo_dir, key, 'info.pkl'))

    ### worker methods
    def load_worker(self, worker_key: str, WorkerClass=None, load_buffer=True):
        """
        loads worker from specified folder
        Args:
            worker_key: folder to grab worker from
            WorkerClass: class to use to load worker (if none, tries to load from class.pkl)
            load_buffer: whether to load replay buffer, if available
        Returns: loaded worker (SB3 algorithm)
        """
        full_dir = os.path.join(self.zoo_dir, worker_key)
        return load_worker(save_dir=full_dir,
                           WorkerClass=WorkerClass,
                           load_buffer=load_buffer)

    def overwrite_worker(self,
                         worker,
                         worker_key: str,
                         save_buffer=True,
                         save_class=True,
                         worker_info=None,
                         ):
        """
        saves worker to folder named worker_key
        Args:
            worker: worker to save
            worker_key: folder to save to
            save_buffer: whether to save replay buffer, if available
            save_class: whether to save class of algorithm (suprised this works)
        """

        full_dir = os.path.join(self.zoo_dir, worker_key)

        return overwrite_worker(worker=worker,
                                worker_info=worker_info,
                                save_dir=full_dir,
                                save_buffer=save_buffer,
                                save_class=save_class,
                                )

    def worker_exists(self, worker_key: str):
        full_dir = os.path.join(self.zoo_dir, worker_key)
        return worker_exists(save_dir=full_dir)

    def _update_off_policy_buffer(self, buffer, local_buffer):
        """
        adds contents of local buffer to buffer
        Args:
            buffer: target to add to
            local_buffer: buffer to draw from
        """
        if local_buffer.full:
            pos_0 = (local_buffer.pos + 1)%local_buffer.buffer_size
        else:
            pos_0 = 0

        for i in range(local_buffer.size()):
            pos = (i + pos_0)%local_buffer.buffer_size
            if isinstance(local_buffer, DictReplayBuffer):
                assert isinstance(buffer, DictReplayBuffer)
                obs = {key: local_buffer.observations[key][pos]
                       for key in local_buffer.observations}
            else:
                obs = local_buffer.observations[pos]
            action = local_buffer.actions[pos]
            reward = local_buffer.rewards[pos]
            done = local_buffer.dones[pos]
            infos = [{'TimeLimit.truncated': True} if timeout else {}
                     for timeout in local_buffer.timeouts[pos]]
            if local_buffer.optimize_memory_usage:
                next_obs = local_buffer.observations[(pos + 1)%local_buffer.buffer_size]
            else:
                if isinstance(local_buffer, DictReplayBuffer):
                    assert isinstance(buffer, DictReplayBuffer)
                    next_obs = {key: local_buffer.next_observations[key][pos]
                                for key in local_buffer.next_observations}
                else:
                    next_obs = local_buffer.next_observations[pos]
            buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                infos=infos,
            )

    def _update_on_policy_buffer(self, buffer, local_buffer):
        """
        adds contents of local buffer to buffer
        Args:
            buffer: target to add to
            local_buffer: buffer to draw from
        """
        if local_buffer.full:
            pos_0 = (local_buffer.pos + 1)%local_buffer.buffer_size
        else:
            pos_0 = 0
        for i in range(local_buffer.size()):
            pos = (i + pos_0)%local_buffer.buffer_size
            if isinstance(local_buffer, DictRolloutBuffer):
                assert isinstance(buffer, DictRolloutBuffer)
                obs = {key: local_buffer.observations[key][pos]
                       for key in local_buffer.observations}
            else:
                obs = local_buffer.observations[pos]
            action = local_buffer.actions[pos]
            reward = local_buffer.rewards[pos]
            episode_start = local_buffer.episode_starts[pos]
            value = torch.tensor(local_buffer.values[pos])
            log_prob = torch.tensor(local_buffer.log_probs[pos])
            if buffer.full:
                # start filling from the front
                buffer.pos = 0
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                episode_start=episode_start,
                value=value,
                log_prob=log_prob,
            )

    def clear_worker_buffer(self, worker_key: str, WorkerClass=None):
        """
        clears a saved worker's buffer
        Args:
            worker_key: folder to load from/save to
            WorkerClass: class to use, if known
        """
        worker, worker_info = self.load_worker(worker_key=worker_key,
                                               load_buffer=False,
                                               WorkerClass=WorkerClass,
                                               )

        self.overwrite_worker(worker=worker,
                              worker_key=worker_key,
                              save_buffer=False,
                              save_class=self.class_is_saved(worker_key=worker_key),
                              worker_info=worker_info,
                              )

    def update_worker_buffer(self, local_worker, worker_key: str, env, WorkerClass=None):
        """
        takes active agent's buffer and stores it into buffer of worker_key
            assumes buffer inherits BaseBuffer
            only works if both both agents share a buffer type (i.e. both OnPolicy or both OffPolicy algorithms)
        Args:
            local_worker: worker to copy from
            worker_key: folder to load from/save to
            WorkerClass: class to use, if known
        """
        worker, worker_info = self.load_worker(worker_key=worker_key,
                                               load_buffer=True,
                                               WorkerClass=WorkerClass,
                                               )
        if isinstance(worker, OffPolicyAlgorithm):
            assert isinstance(local_worker, OffPolicyAlgorithm)
            worker.set_env(env=env)
            assert isinstance(worker, OffPolicy)
            worker.update_from_buffer(local_buffer=local_worker.replay_buffer)
        elif isinstance(worker, OnPolicyAlgorithm):
            assert isinstance(local_worker, OnPolicyAlgorithm)
            worker.set_env(env=env)
            assert isinstance(worker, OnPolicy)
            worker.update_from_buffer(local_buffer=local_worker.rollout_buffer)
        else:
            raise Exception("what algorithigm is this", worker)

        self.overwrite_worker(worker=worker,
                              worker_key=worker_key,
                              save_buffer=True,
                              save_class=self.class_is_saved(worker_key=worker_key),
                              worker_info=worker_info,
                              )
        return worker, worker_info

    ### other methods
    def overwrite_other(self,
                        other,
                        other_key: str,
                        other_info=None,
                        ):
        """
        pickles another object and saves it into key
        Args:
            other: some object
            other_key: folder to save into
            other_info: info to save
        Returns:
        """
        overwrite_other(other=other,
                        save_dir=os.path.join(self.zoo_dir, other_key),
                        other_info=other_info
                        )

    def load_other(self,
                   other_key: str,
                   ):
        """
        loads other object
        Args:
            other_key: folder to load from
        Returns: other object, other info
        """
        return load_other(save_dir=os.path.join(self.zoo_dir, other_key))

    def other_exists(self, other_key: str):
        return other_exists(save_dir=os.path.join(self.zoo_dir, other_key))

    ### misc
    def class_is_saved(self, worker_key: str):
        """
        returns if class is saved in worker key
        Args:
            worker_key: folder to check
        Returns:
            boolean
        """
        return os.path.exists(os.path.join(self.zoo_dir, worker_key, 'class.pkl'))

    ### global methods
    def save(self, save_dir):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        shutil.copytree(self.zoo_dir, save_dir)

    def load(self, save_dir):
        self.clear()
        shutil.copytree(save_dir, self.zoo_dir)
        self.saved_workers = set(os.listdir(self.zoo_dir))

    def clear(self):
        shutil.rmtree(self.zoo_dir)


if __name__ == '__main__':
    import sys
    from unstable_baselines3.ppo.PPO import WorkerPPO
    from stable_baselines3.ppo import MlpPolicy

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
    zoo = ZooCage(zoo_dir=os.path.join(DIR, 'data', 'zoo_cage'))
    for i in range(100):
        zoo.overwrite_worker(worker=WorkerPPO(policy=MlpPolicy, env='CartPole-v1'),
                             worker_key=str(i),
                             )
    save_dir = os.path.join(DIR, 'data', 'zoo_cage_save', 'test', 'test', 'test')
    zoo.save(save_dir=save_dir)
    zoo2 = ZooCage(zoo_dir=os.path.join(DIR, 'data', 'zoo_cage2'))
    zoo2.load(save_dir=save_dir)
    print(len(zoo2.saved_workers))
    zoo.clear()
    zoo2.clear()
    shutil.rmtree(save_dir)
