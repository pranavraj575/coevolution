import torch
from torch.utils.data import Dataset
from torchrl.data import ReplayBuffer, LazyMemmapStorage, ListStorage
import os, sys, shutil, pickle


class RBConstantSizeDataset(Dataset):
    """
    this will only work for inputs of the same size
    """

    def __init__(self, capacity=int(1e6), from_file=None, STORAGE=LazyMemmapStorage):
        self.memory = ReplayBuffer(storage=STORAGE(max_size=capacity), )
        self.capacity = capacity
        self.tensor_len = None

    def push(self, item):
        self.memory.add(item)

    def extend(self, items):
        self.memory.extend(items)

    def save(self, dirname):
        if os.path.exists(dirname):
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)
            else:
                os.remove(dirname)
        self.memory.dumps(dirname)

    def load(self, dirname):
        if self.tensor_len is None:
            raise Exception("ERROR: TO LOAD THIS REPRESENTATION, tensor_len must be specified on intialization")
        self.memory.add(torch.zeros(size=(self.tensor_len,)))  # intialize the memory size
        self.memory.empty()  # delete the dummy tensor
        self.memory.loads(dirname)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory.__getitem__(item)


class ReplayBufferDiskStorage:
    def __init__(self, storage_dir, capacity=1e6, ):
        self.idx = 0
        self.size = 0
        self.storage_dir = storage_dir
        self.capacity = capacity
        self.reset_storage()

    def close(self):
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)

    def reset_storage(self):
        self.close()
        os.makedirs(self.storage_dir)
        self.size = 0
        self.idx = 0

    def _get_file(self, idx):
        return os.path.join(self.storage_dir, str(idx) + '.pkl')

    def extend(self, items):
        for item in items:
            self.push(item)

    def push(self, item):
        pickle.dump(item, open(self._get_file(self.idx), 'wb'))

        self.size = max(self.idx + 1, self.size)
        self.idx = (self.idx + 1)%self.capacity

    def _grab_item_by_idx(self, idx):
        return pickle.load(open(self._get_file(idx=idx), 'rb'))

    def sample_one(self):
        return self[torch.randint(0, self.size, (1,))]

    def sample(self, batch):
        for _ in range(batch):
            yield self.sample_one()

    def __getitem__(self, item):
        if item >= self.size:
            raise IndexError
        return self._grab_item_by_idx(idx=int((self.idx + item)%self.size))

    def __len__(self):
        return self.size


if __name__ == '__main__':
    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
    test = ReplayBufferDiskStorage(capacity=3, storage_dir=os.path.join(DIR, 'data', 'buffers', 'replay_buffer_test'))
    test.extend('help')
    print(list(test.sample(3)))
