import numpy as np
import torch


class CoevolutionBase:
    """
    general coevolution algorithm
    """

    def __init__(self, population_sizes, num_teams=2):
        """
        Args:
            population_sizes: list of number of agents in each population
                usually can just be a list of one element
                multiple populations are useful if there are different 'types' of agents
                    in the game that take different inputs/have access to different actions
            num_teams: number of teams in the game, default 2
        """
        self.population_sizes = population_sizes
        self.num_teams = num_teams
        self.N = sum(self.population_sizes)
        if self.N%num_teams != 0:
            print("WARNING: number of agents is not divisible by num teams")
            print('\tthis is fine, but will have non-uniform game numbers for each agent')

    def index_to_pop_index(self, i):
        """
        returns the index of agent specified by i
        Args:
            i: index
        Returns:
            population index (which population), index in population
        """
        pop_idx = 0
        while i - self.Ns[pop_idx] >= 0:
            i -= self.Ns[pop_idx]
            pop_idx += 1
        return pop_idx, i

    def create_random_captians(self, N=None):
        """
        Args:
            N: if None, uses self.N
        Returns: iterable of edges (i,j) that is a matching on (0,...,N)
            if N is not divisible by number of teams, a set of agents are chosen twice
            othewise, each agent is chosen once
        """

        if N is None:
            N = self.N
        unused = set(range(N))

        while len(unused) > self.num_teams:
            choice = np.random.choice(list(unused), self.num_teams, replace=False)
            for i in choice:
                unused.remove(i)
            yield tuple(choice)
        if unused:
            remaining = self.num_teams - len(unused)
            # randomly sort unused, and concatenate with a random choice of the other agent indices
            yield (tuple(np.random.choice(list(unused), len(unused), replace=False)) +
                   tuple(np.random.choice(list(set(range(N)).difference(unused)), remaining, replace=False)))

    def epoch(self):
        # TODO: parallelizable
        for choice in self.create_random_captians():
            self.train_and_update_results(choice)
        self.terminate_and_clone()

    def terminate_and_clone(self):
        """
        terminates poorly performing agents and replaces them with clones
            can also replace through sexual reproduction, if that is possible

        also updates internal variables to reflect the new agents
        """
        raise NotImplementedError

    def train_and_update_results(self, captian_choices):
        """
        takes a choice of team captians and trains them in RL environment
        updates variables to reflect result of game(s)
        Args:
            captian_choices: tuple of self.num_teams indices
        """
        raise NotImplementedError


class TwoPlayerAdversarialCoevolution(CoevolutionBase):
    def __init__(self, population_size, outcome_fn, clone_fn, elo_update=32, init_tau=1.):
        """
        Args:
            population_size: size of population to train
            outcome_fn: function that takes two captain indices and trains them in a single game against each other
                outcome_fn(player idx,opponent idx)
                    returns 1 for win, 0 for tie, -1 for loss
            clone_fn: takes in (original list, replacement list)
                replaces agents in the original indices with respective replacement
            elo_update: k for updating elos
            init_tau: initial temperature for deleting agents
        """
        super().__init__(population_sizes=[population_size], num_teams=2)
        self.outcome_fn = outcome_fn
        self.clone_fn = clone_fn

        # keeps track of number of games and record for each agent
        self.captian_fitness = np.ones(self.N)*1000  # ELO
        self.captian_wins = np.zeros(self.N)
        self.captian_losses = np.zeros(self.N)
        self.captian_ties = np.zeros(self.N)
        self.captian_elos = np.ones(self.N)*1000
        self.elo_update = elo_update
        self.tau = init_tau

    def rescale_elos(self, base_elo=1000.):
        """
        scales all elos so that base_elo is average
            does not change any elo calculations, as they are all based on relative difference
        Args:
            base_elo: elo to make the 'average' elo
        """
        self.captian_elos += base_elo - np.sum(self.captian_elos)/self.N

    def train_and_update_results(self, captian_choices):
        i, j = captian_choices
        outcome = self.outcome_fn(i, j)
        if outcome == 0:
            for idx in captian_choices:
                self.captian_ties[idx] += 1
        else:
            winner, loser = (i, j) if outcome == 1 else (j, i)
            self.captian_wins[winner] += 1
            self.captian_losses[loser] += 1
        obs_prob_i = (outcome + 1)/2  # observed probability of i win, .5 if tie, 1 if win, 0 if loss
        expected_prob_i = 1/(1 + np.power(10, -(self.captian_elos[i] - self.captian_elos[j])/400))
        self.captian_elos[i] += self.elo_update*(obs_prob_i - expected_prob_i)
        self.captian_elos[j] += self.elo_update*((1 - obs_prob_i) - (1 - expected_prob_i))

    def terminate_and_clone(self):
        dist = torch.softmax(torch.tensor(self.captian_elos/self.tau), dim=-1)
        self.clone_fn(torch.arange(self.N),torch.multinomial(dist,self.N))


if __name__ == '__main__':
    popsize=10
    agents=torch.arange(popsize)%3
    def clone_fn(original,replacements):
        agents[original]=agents[replacements]
    test = TwoPlayerAdversarialCoevolution(population_size=popsize,
                                           outcome_fn=lambda i, j: max(i, j),
                                           clone_fn=clone_fn,
                                           )
    print(agents)
    test.terminate_and_clone()
    print(agents)
    quit()
    test.captian_elos[0] = 0
    test.rescale_elos(10)
    print(test.captian_elos)
    print(np.mean(test.captian_elos))

    test = CoevolutionBase(population_sizes=[30], num_teams=2)
    N = test.N
    matching = list(test.create_random_captians(N=N))
    tracker = np.zeros(N)
    for edge in matching:
        print(edge)
        for i in edge:
            tracker[i] += 1
    print(tracker)
