from pettingzoo import AECEnv, ParallelEnv
from multi_agent_algs.parallel_alg import ParallelAlgorithm
from multi_agent_algs.aec_alg import AECAlgorithm
from typing import Union


def multi_agent_algorithm(
        env: Union[AECEnv, ParallelEnv],
        workers,
        *args,
        **kwargs, ) -> Union[AECAlgorithm, ParallelAlgorithm]:
    if isinstance(env, AECEnv):
        return AECAlgorithm(env=env, workers=workers, *args, **kwargs)
    elif isinstance(env, ParallelEnv):
        return ParallelAlgorithm(parallel_env=env, workers=workers, *args, **kwargs)
    else:
        raise Exception('env not recognized:', env)
