from multiprocessing import Pool
import time
import os
from matplotlib import pyplot as plt


class Test():
    processes = 1

    def _split_train_choice(self, cap_unique):
        return None

    def test(self):
        captian_and_unique_choices =[1]
        if self.processes >= 1:
            # TODO: This does not work for some reason
            with Pool(processes=self.processes) as pool:
                all_items_to_save = pool.map(self._split_train_choice, captian_and_unique_choices)
            print(all_items_to_save)

print(Test().test())


max_cores = len(os.sched_getaffinity(0))


def f(x):
    i = 0
    tim = time.time()
    for _ in range(x):
        i += 1
    return time.time() - tim


add_ops = 316900
runtimes = []
seq_runtimes = []
for processes in range(1, max_cores*3):
    with Pool(processes=processes) as pool:
        # print "[0, 1, 4,..., 81]"
        tim = time.time()
        print(pool.map(f, [add_ops for _ in range(processes)]))
        runtimes.append(time.time() - tim)

    with Pool(processes=1) as pool:
        # print "[0, 1, 4,..., 81]"
        tim = time.time()
        print(pool.map(f, [add_ops for _ in range(processes)]))
        seq_runtimes.append(time.time() - tim)

plt.plot(runtimes, label='runtimes')
plt.plot(seq_runtimes, label='seq runtimes')
plt.plot((max_cores, max_cores), plt.ylim(), label='max cores')
plt.ylabel('time')
plt.xlabel('processes (p)')
plt.title('time to run p parallel sets of add operations with p processes')
plt.legend()
plt.savefig('data/temp.png')
plt.show()
