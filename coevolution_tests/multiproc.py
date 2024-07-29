
from pathos.pools import ProcessPool as Pool
import time
import os
from matplotlib import pyplot as plt


def f(x):
    i = 0
    tim = time.time()
    for _ in range(x):
        i += 1
    return time.time() - tim


max_cores = len(os.sched_getaffinity(0))
add_ops = 316900
runtimes = []
seq_runtimes=[]
for processes in range(1, max_cores* 3):
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

plt.plot(runtimes,label='runtimes')
plt.plot(seq_runtimes,label='seq runtimes')
plt.plot((max_cores,max_cores),plt.ylim(),label='max cores')
plt.ylabel('time')
plt.xlabel('processes (p)')
plt.title('time to run p parallel sets of add operations with p processes')
plt.legend()
plt.savefig('data/temp.png')
plt.show()