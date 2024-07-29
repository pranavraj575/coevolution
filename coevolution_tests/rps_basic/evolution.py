import torch
from src.coevolver import CaptianCoevolution
from coevolution_tests.rps_basic.game import plot_dist_evolution, SingleOutcome

if __name__ == '__main__':
    import os, sys
    from src.team_trainer import TeamTrainer

    torch.random.manual_seed(69)

    # print(outcomes((torch.tensor([[0], [1], [2], [0]]), torch.tensor(([[1], [0], [0], [0]]))))[-1])

    DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))
    plot_dir = os.path.join(DIR, 'data', 'plots', 'tests_rps_evolution')
    if not os.path.exists((plot_dir)):
        os.makedirs(plot_dir)

    popsize = 100
    agents = torch.randint(0, 3, (popsize,))


    def clone(init, replacement):
        global agents
        temp = agents.clone()
        agents[init] = temp[replacement]


    def mutate():
        global agents
        idc = torch.randint(0, len(agents), (1,))
        agents[idc] = torch.randint(0, 3, (1,))
        return idc


    trainer = CaptianCoevolution(population_sizes=[popsize],
                                 outcome_fn_gen=lambda: SingleOutcome(agents=agents),
                                 team_trainer=TeamTrainer(num_agents=popsize),
                                 clone_fn=clone,
                                 mutation_fn=mutate,
                                 elo_update=.05
                                 )

    init_dists = []
    for epoch in range(100):
        for i in range(2):
            trainer.epoch(rechoose=False)
        print('epoch', epoch)
        trainer.breed()
        trainer.mutate()
        init_distribution = [len(torch.where(agents == i)[0])/popsize for i in range(3)]
        init_dists.append(init_distribution)

        if not (epoch + 1)%10:
            plot_dist_evolution(init_dists, save_dir=os.path.join(plot_dir, 'init_dist.png'))
