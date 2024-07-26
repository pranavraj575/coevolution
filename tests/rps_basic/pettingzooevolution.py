import torch
from src.coevolver import PettingZooCaptianCoevolution
from tests.rps_basic.game import plot_dist_evolution, SingleZooOutcome
from src.utils.dict_keys import *

if __name__ == '__main__':
    import os, sys
    from src.team_trainer import TeamTrainer

    torch.random.manual_seed(69)

    # print(outcomes((torch.tensor([[0], [1], [2], [0]]), torch.tensor(([[1], [0], [0], [0]]))))[-1])

    DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))
    plot_dir = os.path.join(DIR, 'data', 'plots', 'tests_rps_zooevolution')
    if not os.path.exists((plot_dir)):
        os.makedirs(plot_dir)

    popsizes = [1, 1, 28]
    popsize = sum(popsizes)

    trainer = PettingZooCaptianCoevolution(population_sizes=popsizes,
                                           outcome_fn=SingleZooOutcome(),
                                           team_trainer=TeamTrainer(num_agents=popsize),
                                           worker_constructors=[lambda i: (1, {DICT_IS_WORKER: False,
                                                                               DICT_CLONABLE: True,
                                                                               DICT_CLONE_REPLACABLE: False,
                                                                               DICT_TRAIN: False,
                                                                               }),
                                                                lambda i: (2, {DICT_IS_WORKER: False,
                                                                               DICT_CLONABLE: True,
                                                                               DICT_CLONE_REPLACABLE: False,
                                                                               DICT_TRAIN: False,
                                                                               }),
                                                                lambda i: (0, {DICT_IS_WORKER: False,
                                                                               DICT_TRAIN: False,
                                                                               })],
                                           zoo_dir=os.path.join(DIR, 'data', 'test', 'rps_basic_zoo'),
                                           env_constructor=None,
                                           )
    for cage_idx, pop in enumerate(popsizes):
        for i in range(pop):
            animal, info = trainer.zoo[cage_idx].load_animal(key=str(i))

    init_dists = []
    for epoch in range(2000):
        for i in range(10):
            trainer.epoch(rechoose=False)
        print('epoch', epoch)
        trainer.breed()
        trainer.mutate()
        animals = []
        for cage_idx, pop in enumerate(popsizes):
            for i in range(pop):
                animal, info = trainer.zoo[cage_idx].load_animal(key=str(i))
                animals.append(animal)
        init_distribution = [animals.count(i)/popsize for i in range(3)]
        init_dists.append(init_distribution)

        if not (epoch + 1)%10:
            plot_dist_evolution(init_dists, save_dir=os.path.join(plot_dir, 'init_dist.png'))
