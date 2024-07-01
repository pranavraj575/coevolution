import numpy as np


class CoevolutionBase:
    """
    general coevolution algorithm
    """

    def __init__(self, populations, num_teams=2):
        """
        Args:
            populations: list of lists of agents
                usually can just be [[population]]
                multiple populations are useful if there are different 'types' of agents
                    in the game that take different inputs/have access to different actions
            num_teams: number of teams in the game, default 2
        """
        self.populations = populations
        self.num_teams = num_teams
        self.Ns = [len(population) for population in self.populations]
        self.N = sum(self.Ns)
        if self.N%num_teams != 0:
            print("WARNING: number of agents is not divisible by num teams")
            print('\tthis is fine, but will have non-uniform game numbers for each agent')

        # keeps track of number of games and record for each agent
        self.captian_fitness = np.zeros(self.N)
        self.captian_info = [dict() for _ in range(self.N)]

    def index_to_agent(self, i):
        """
        returns the agent specified by i
        Args:
            i: index
        Returns:
            member of self.populations that corresponds to i
        """
        pop_idx = 0
        while i - self.Ns[pop_idx] >= 0:
            i -= self.Ns[pop_idx]
            pop_idx += 1
        return self.populations[pop_idx][i]

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

    def train_and_update_results(self, choice):
        """
        takes a choice of team captians and trains them in RL environment
        updates variables to reflect result of game(s)
        Args:
            choice: tuple of self.num_teams indices
        """
        raise NotImplementedError


if __name__ == '__main__':
    test = CoevolutionBase(populations=[list(range(100))], num_teams=6)
    N = test.N
    matching = list(test.create_random_captians(N=N))
    tracker = np.zeros(N)
    for edge in matching:
        print(edge)
        for i in edge:
            tracker[i] += 1
    print(tracker)
