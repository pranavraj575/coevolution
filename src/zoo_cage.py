import torch, os, pickle, shutil
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import BaseBuffer, DictReplayBuffer, DictRolloutBuffer
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl


class ZooCage:
    def __init__(self, zoo_dir, overwrite_zoo=True):
        self.zoo_dir = zoo_dir
        if not os.path.exists(self.zoo_dir):
            os.makedirs(self.zoo_dir)
        elif overwrite_zoo:
            self.kill_zoo()
            os.makedirs(self.zoo_dir)
        self.active_workers = dict()
        self.saved_workers = set(os.listdir(self.zoo_dir))

    def activate_worker(self, agent_key, worker_key, WorkerClass=None, load_buffer=True):
        """
        loads worker and adds to active workers
        Args:
            agent_key: key to save worker as in dictionary (usually same as pettingzoo enviornment agent id)
            worker_key: folder to grab worker from
            WorkerClass: class to use to load worker (if none, tries to load from class.pkl)
            load_buffer: whether to load replay buffer, if available
        """
        self.active_workers[agent_key] = self.load_worker(worker_key=worker_key,
                                                          WorkerClass=WorkerClass,
                                                          load_buffer=load_buffer,
                                                          )

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
        if WorkerClass is None:
            f = open(os.path.join(full_dir, 'class.pkl'), 'rb')
            WorkerClass = pickle.load(f)
            f.close()
        worker = WorkerClass.load(os.path.join(full_dir, 'worker'))
        if load_buffer:
            if isinstance(worker, OffPolicyAlgorithm):
                buff_file = os.path.join(full_dir, 'replay_buffer.pkl')
                if os.path.exists(buff_file):
                    worker.load_replay_buffer(buff_file)
                else:
                    print('buffer file not found:', buff_file)
                    print('change saving settings perhaps, unless this was intended')
            elif isinstance(worker, OnPolicyAlgorithm):
                worker.rollout_buffer = load_from_pkl(os.path.join(full_dir, 'rollout_buffer.pkl'))
                worker.rollout_buffer.device = worker.device
            else:
                print('WHAT ALGORITHM IS THIS', worker)
        return worker

    def save_active_worker(self, agent_key, worker_key: str, save_buffer=True, save_class=True):
        """
        saves worker to folder named worker_key
        Args:
            agent_key: key of agent in active_worker dict (usually something like 'player 0' in environment)
            worker_key: worker to save, folder to save to
            save_buffer: whether to save replay buffer, if available
            save_class: whether to save class of algorithm (suprised this works)
        """
        self.overwrite_worker(worker=self.active_workers[agent_key],
                              worker_key=worker_key,
                              save_buffer=save_buffer,
                              save_class=save_class,
                              )

    def overwrite_worker(self, worker, worker_key: str, save_buffer=True, save_class=True):
        """
        saves worker to folder named worker_key
        Args:
            worker: worker to save
            worker_key: folder to save to
            save_buffer: whether to save replay buffer, if available
            save_class: whether to save class of algorithm (suprised this works)
        """
        full_dir = os.path.join(self.zoo_dir, worker_key)

        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        # worker inherits save method from stable baselines
        worker.save(os.path.join(full_dir, 'worker'))
        if save_buffer:
            if isinstance(worker, OffPolicyAlgorithm):
                # worker has a replay buffer that should also probably be saved
                worker.save_replay_buffer(os.path.join(full_dir, 'replay_buffer.pkl'))
            elif isinstance(worker, OnPolicyAlgorithm):
                save_to_pkl(os.path.join(full_dir, 'rollout_buffer.pkl'), worker.rollout_buffer)
            else:
                print('WHAT ALGORITHM IS THIS', worker)
        if save_class:
            cls = type(worker)
            f = open(os.path.join(full_dir, 'class.pkl'), 'wb')
            pickle.dump(cls, f)
            f.close()
        self.saved_workers.add(worker_key)

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

    def update_worker_buffer(self, agent_key, worker_key: str, WorkerClass=None):
        """
        takes active agent's buffer and stores it into buffer of worker_key
            assumes buffer inherits BaseBuffer
        Args:
            agent_key: agent to copy from
            worker_key: folder to load from/save to
            WorkerClass: class to use, if known
        """

        worker = self.load_worker(worker_key=worker_key, load_buffer=True, WorkerClass=WorkerClass)
        local_worker = self.active_workers[agent_key]
        if isinstance(worker, OffPolicyAlgorithm):
            assert isinstance(local_worker, OffPolicyAlgorithm)
            self._update_off_policy_buffer(buffer=worker.replay_buffer,
                                           local_buffer=local_worker.replay_buffer,
                                           )

        elif isinstance(worker, OnPolicyAlgorithm):
            assert isinstance(local_worker, OnPolicyAlgorithm)

            self._update_on_policy_buffer(buffer=worker.rollout_buffer,
                                          local_buffer=local_worker.rollout_buffer,
                                          )
        else:
            raise Exception("what algorithigm is this", worker)

        self.overwrite_worker(worker=worker,
                              worker_key=worker_key,
                              save_buffer=True,
                              save_class=self.class_is_saved(worker_key=worker_key),
                              )
        return worker

    def class_is_saved(self, worker_key: str):
        """
        returns if class is saved in worker key
        Args:
            worker_key: folder to check
        Returns:
            boolean
        """
        return os.path.exists(os.path.join(self.zoo_dir, worker_key, 'class.pkl'))

    def save_zoo(self, save_dir):
        shutil.rmtree(save_dir)
        shutil.copytree(self.zoo_dir, save_dir)

    def load_zoo(self, save_dir):
        self.kill_zoo()
        shutil.copytree(save_dir, self.zoo_dir)
        self.saved_workers = set(os.listdir(self.zoo_dir))

    def kill_zoo(self):
        shutil.rmtree(self.zoo_dir)


if __name__ == '__main__':
    import sys
    from parallel_algs.ppo.PPO import WorkerPPO
    from stable_baselines3.ppo import MlpPolicy

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
    zoo = ZooCage(zoo_dir=os.path.join(DIR, 'data', 'zoo_cage'))
    for i in range(100):
        zoo.overwrite_worker(worker=WorkerPPO(policy=MlpPolicy, env='CartPole-v0'),
                             worker_key=str(i),
                             )
    save_dir = os.path.join(DIR, 'data', 'zoo_cage_save')
    zoo.save_zoo(save_dir=save_dir)
    zoo2 = ZooCage(zoo_dir=os.path.join(DIR, 'data', 'zoo_cage2'))
    zoo2.load_zoo(save_dir=save_dir)
    print(len(zoo2.saved_workers))
    zoo.kill_zoo()
    zoo2.kill_zoo()
    shutil.rmtree(save_dir)
