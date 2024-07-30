import os, pickle, shutil
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl


def overwrite_worker(worker, worker_info, save_dir, save_buffer=True, save_class=True):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # save info
    if worker_info is None:
        worker_info = dict()
    info_file = os.path.join(save_dir, 'info.pkl')
    f = open(info_file, 'wb')
    pickle.dump(worker_info, f)
    f.close()

    # save class
    if save_class:
        cls = type(worker)
        f = open(os.path.join(save_dir, 'class.pkl'), 'wb')
        pickle.dump(cls, f)
        f.close()

    # save buffer
    if save_buffer:
        if isinstance(worker, OffPolicyAlgorithm):
            worker.save_replay_buffer(os.path.join(save_dir, 'replay_buffer.pkl'))
        elif isinstance(worker, OnPolicyAlgorithm):
            save_to_pkl(os.path.join(save_dir, 'rollout_buffer.pkl'), worker.rollout_buffer)
        else:
            raise Exception('type not valid', type(worker))
    # save worker
    worker.save(os.path.join(save_dir, 'worker.zip'))


def load_worker(save_dir, WorkerClass=None, load_buffer=True):
    # load info
    filename = os.path.join(save_dir, 'info.pkl')
    if os.path.exists(filename):
        f = open(filename, 'rb')
        worker_info = pickle.load(f)
        f.close()
    else:
        worker_info = {}

    # load class
    if WorkerClass is None:
        f = open(os.path.join(save_dir, 'class.pkl'), 'rb')
        WorkerClass = pickle.load(f)
        f.close()

    # load worker
    worker = WorkerClass.load(os.path.join(save_dir, 'worker.zip'))

    # load buffer
    if load_buffer:
        if isinstance(worker, OffPolicyAlgorithm):
            buff_file = os.path.join(save_dir, 'replay_buffer.pkl')
            if os.path.exists(buff_file):
                worker.load_replay_buffer(buff_file)
        elif isinstance(worker, OnPolicyAlgorithm):
            buff_file = os.path.join(save_dir, 'rollout_buffer.pkl')
            if os.path.exists(buff_file):
                worker.rollout_buffer = load_from_pkl(buff_file)
                worker.rollout_buffer.device = worker.device
        else:
            print('WHAT ALGORITHM IS THIS', worker)

    return worker, worker_info


def worker_exists(save_dir):
    return (os.path.exists(os.path.join(save_dir, 'info.pkl')) and
            os.path.exists(os.path.join(save_dir, 'worker.zip')))
