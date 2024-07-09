import torch
from src.coevolver import TwoPlayerAdversarialCoevolution
from rps_basic_game import plot_dist_evolution, single_game_outcome


def double_game_outcome(i, j, agents):
    a, b = agents[(i, j),]

    if a < 3 and b < 3:
        return single_game_outcome(i, j, agents)
    if a >= 3 and b >= 3:
        # ['RR', 'PP', 'SS', 'RP', 'RS', 'PS']
        # mapped to
        # ['PS', 'RS', 'RP', ...]
        # equivalent to
        # [ R P S ]
        return single_game_outcome(i, j, agents=6 - agents)
    elif ({a, b} in ({0, 4}, {1, 3}, {2, 5})):
        if a < b:
            return 1
        else:
            return -1
    else:
        if a > b:
            return 1
        else:
            return -1


if __name__ == '__main__':
    import os, sys

    torch.random.manual_seed(69)

    # print(outcomes((torch.tensor([[0], [1], [2], [0]]), torch.tensor(([[1], [0], [0], [0]]))))[-1])

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
    plot_dir = os.path.join(DIR, 'data', 'plots', 'tests_rps2_evolution')
    if not os.path.exists((plot_dir)):
        os.makedirs(plot_dir)

    popsize = 100
    agents = torch.randint(0, 6, (popsize,))


    def clone(init, replacement):
        global agents
        temp = agents.clone()
        agents[init] = temp[replacement]


    def mutate(p=.01):
        global agents
        mask = torch.rand(popsize) < p
        agents[mask] = torch.randint(0, 6, (torch.sum(mask).item(),))


    trainer = TwoPlayerAdversarialCoevolution(population_size=popsize,
                                              outcome_fn=lambda i, j: double_game_outcome(i, j, agents=agents),
                                              clone_fn=clone,
                                              init_tau=1200,
                                              mutation_fn=mutate
                                              )

    init_dists = []
    for epoch in range(1000):
        for i in range(100):
            trainer.epoch(rechoose=False)
        trainer.breed()
        trainer.mutate()
        init_distribution = [len(torch.where(agents == i)[0])/popsize for i in range(6)]
        init_dists.append(init_distribution)

        if not (epoch + 1)%10:
            plot_dist_evolution(init_dists, save_dir=os.path.join(plot_dir, 'init_dist.png'),
                                labels=['RR', 'PP', 'SS', 'RP', 'RS', 'PS'])
