import numpy as np
import torch
from src.team_trainer import TeamTrainer
from src.game_outcome import PlayerInfo, OutcomeFn
import time


class CoevolutionBase:
    """
    general coevolution algorithm
    """

    def __init__(self, outcome_fn: OutcomeFn, population_sizes, team_sizes=(1, 1)):
        """
        Args:
            outcome_fn: collect outcomes and do RL training
            population_sizes: list of number of agents in each population
                usually can just be a list of one element
                multiple populations are useful if there are different 'types' of agents
                    in the game that take different inputs/have access to different actions
            team_sizes: tuple of number of agents in each team
                i.e. (1,1) is a 1v1 game
        """
        self.outcome_fn = outcome_fn
        self.population_sizes = population_sizes
        self.team_sizes = team_sizes
        self.num_teams = len(team_sizes)
        self.N = sum(self.population_sizes)
        if self.N%self.num_teams != 0:
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
        while i - self.population_sizes[pop_idx] >= 0:
            i -= self.population_sizes[pop_idx]
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

    def epoch(self, rechoose=True):
        # TODO: parallelizable
        for choice in self.create_random_captians():
            self.train_and_update_results(choice)
        if rechoose:
            self.breed()
            self.mutate()

    def breed(self):
        """
        terminates poorly performing agents and replaces them with clones
            can also replace through sexual reproduction, if that is possible

        also updates internal variables to reflect the new agents
        """
        raise NotImplementedError

    def mutate(self):
        pass

    def train_and_update_results(self, captian_choices):
        """
        takes a choice of team captians and trains them in RL environment
        updates variables to reflect result of game(s)
        Args:
            captian_choices: tuple of self.num_teams indices
        """
        raise NotImplementedError


class CaptianCoevolution(CoevolutionBase):
    """
    coevolution where matches are played between team captains
        i.e. a team built around team captians  (t_1,t_2,...) (one for each team) will play each other
            the score of the game will be used to update the team captian's elo
        this way, the performance of a team built to support the captian will only update the elo of the captian

        in teams with 1 member, this is not relevant, and is simple coevolution
    """
    def __init__(self,
                 outcome_fn: OutcomeFn,
                 population_sizes,
                 team_trainer: TeamTrainer,
                 clone_fn=None,
                 team_sizes=(1, 1),
                 elo_conversion=400/np.log(10),
                 elo_update=32*np.log(10)/400,
                 mutation_fn=None,
                 noise_model=None,
                 ):
        """
        Args:
            outcome_fn: handles training agents and returning game results
            population_sizes:
            team_trainer: model to select teams given a captian, may be trained alongside coevolution
                in games with 1 captain, this mostly does nothing, and can just be the default TeamTrainer class
            team_sizes:
            clone_fn:
            elo_conversion: to keep calculations simple, we use a scaled version of elo where the probability of
                player a (rating Ra) winning against player b (rating Rb) is simply 1/(1+e^{Rb-Ra})
                    i.e. win probabilities are calculated with softmax([Ra,Rb])
                to convert between this and classic elo (where win probs are 1/(1+10^{(Rb'-Ra')/400}), we simply scale
                    by the factor 400/np.log(10)
                This value is only used for display purposes, and will not change behavior of the learning algorithm
            elo_update: Ra'=Ra+elo_update*(Sa-Ea) (https://en.wikipedia.org/wiki/Elo_rating_system).
                We are using scaled elo, so keep this in mind.
                    a 'classic' elo update of 32 is equivalent to a scaled 32*log(10)/400
            mutation_fn:
            noise_model: model to use to set noise in team selectoin
                takes T-element multinomial distributions and returns another (with added noise)
                    ((N,T) -> (N,T))
                updated with set_noise_model

        """
        super().__init__(outcome_fn=outcome_fn,
                         population_sizes=population_sizes,
                         team_sizes=team_sizes,
                         )
        self.team_trainer = team_trainer
        self.noise_model = noise_model
        self.clone_fn = clone_fn
        self.captian_elos = torch.zeros(self.N)
        self.elo_update = elo_update
        self.mutation_fn = mutation_fn
        self.elo_conversion = elo_conversion

    def set_noise_model(self, noise_model):
        self.noise_model = noise_model

    def train_and_update_results(self, captian_choices):
        """
        takes a choice of team captians and trains them in RL environment
        updates variables to reflect result of game(s)
        Args:
            captian_choices: tuple of self.num_teams indices
        """
        teams = []
        captian_positions = []
        # expected win probabilities, assuming the teams win probability is determined by captian elo
        expected_win_probs = torch.softmax(self.captian_elos[captian_choices,],
                                           dim=-1)
        for captian, team_size in zip(captian_choices, self.team_sizes):
            team = self.team_trainer.add_member_to_team(member=captian,
                                                        T=team_size,
                                                        N=1,
                                                        noise_model=self.noise_model,
                                                        )
            pos = torch.where(team.view(-1) == captian)[0].item()
            captian_positions.append(pos)
            team = self.team_trainer.fill_in_teams(initial_teams=team,
                                                   noise_model=self.noise_model,
                                                   )
            teams.append(team)
        team_outcomes = self.outcome_fn.get_outcome(team_choices=teams,
                                                    train=None,
                                                    )
        for (captian,
             team,
             (team_outcome, player_infos),
             expected_outcome) in zip(captian_choices,
                                      teams,
                                      team_outcomes,
                                      expected_win_probs):
            self.captian_elos[captian] += self.elo_update*(team_outcome - expected_outcome)
            for player_info in player_infos:
                player_info: PlayerInfo
                self.team_trainer.add_to_buffer(scalar=team_outcome,
                                                obs_preembed=player_info.obs_preembed,
                                                team=team,
                                                obs_mask=player_info.obs_mask,
                                                )

    def breed(self):
        if self.clone_fn is None:
            return
        dist = torch.softmax(self.captian_elos, dim=-1)
        # sample from this distribution with replacements
        replacements = torch.multinomial(dist, self.N, replacement=True)
        self.clone_fn(torch.arange(self.N), replacements)
        for arr in (self.captian_elos,):
            temp_arr = arr.clone()
            arr[np.arange(self.N)] = temp_arr[replacements]
        self.rebase_elos()

    def mutate(self):
        if self.mutation_fn is not None:
            idc = self.mutation_fn()
        else:
            super().mutate()

    def _get_rebased_elos(self, elos, base_elo):
        """
        scales all elos so that base_elo is average
        Args:
            elos: elos to rescale
            base_elo: elo to make the 'average' elo
        Returns: rebased elos
        """
        return elos + (base_elo - torch.mean(elos))

    def rebase_elos(self, base_elo=0.):
        """
        scales all elos so that base_elo is average
            does not change any elo calculations, as they are all based on relative difference
        Args:
            base_elo: elo to make the 'average' elo
        """
        self.captian_elos = self._get_rebased_elos(elos=self.captian_elos, base_elo=base_elo)

    def get_classic_elo(self, base_elo=None):
        """
        gets 'classic' elo values
        Args:
            base_elo: if specified, rebases the elos so this is the average elo
        Returns:
            'classic' elos, such that for players with 'classic' elos Ra', Rb', the win probability of a is
                1/(1+e^{(Rb'-Ra')/self.elo_conversion})
            i.e. if the default elo_conversion is used, this is  1/(1+10^{(Rb'-Ra')/400}), the standard value
        """

        scaled_elos = self.captian_elos*self.elo_conversion
        if base_elo is not None:
            scaled_elos = self._get_rebased_elos(elos=scaled_elos, base_elo=base_elo)
        return scaled_elos


class PettingZooCaptianCoevolution(CaptianCoevolution):
    """
    keeps track of a set of stable baseline algorithms, and uses them to
    """
    def __init__(self,
                 env_constructor,
                 outcome_fn: OutcomeFn,
                 population_sizes,
                 team_trainer: TeamTrainer,
                 team_sizes=(1, 1),
                 elo_update=1,
                 ):
        super().__init__(outcome_fn=outcome_fn,
                         population_sizes=population_sizes,
                         team_trainer=team_trainer,
                         team_sizes=team_sizes,
                         elo_update=elo_update,
                         mutation_fn=None,
                         clone_fn=None,
                         )
        raise NotImplementedError
        self.env_constructor = env_constructor

    def breed(self):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

class TwoTeamsCaptainCoevolution(CoevolutionBase):
    def __init__(self,
                 population_size,
                 outcome_fn,
                 clone_fn,
                 elo_update=32,
                 init_tau=100.,
                 mutation_fn=None,
                 ):
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
            mutation_fn: mutate agents
        """
        super().__init__(population_sizes=[population_size], num_teams=2)
        self.outcome_fn = outcome_fn
        self.clone_fn = clone_fn
        self.mutation_fn = mutation_fn

        # keeps track of number of games and record for each agent
        self.elo_update = elo_update
        self.tau = init_tau
        self.reset_vals()

    def reset_vals(self):
        self.captian_elos = np.ones(self.N)*1000

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

        obs_prob_i = (outcome + 1)/2  # observed probability of i win, .5 if tie, 1 if win, 0 if loss
        expected_prob_i = 1/(1 + np.power(10, -(self.captian_elos[i] - self.captian_elos[j])/400))
        self.captian_elos[i] += self.elo_update*(obs_prob_i - expected_prob_i)
        self.captian_elos[j] += self.elo_update*((1 - obs_prob_i) - (1 - expected_prob_i))

    def mutate(self):
        if self.mutation_fn is not None:
            self.mutation_fn()

    def breed(self):
        dist = torch.softmax(torch.tensor(self.captian_elos/self.tau), dim=-1)
        # sample from this distribution with replacements
        replacements = torch.multinomial(dist, self.N, replacement=True)
        self.clone_fn(torch.arange(self.N), replacements)
        for arr in (self.captian_elos,):
            temp_arr = arr.copy()
            arr[np.arange(self.N)] = temp_arr[replacements]
        self.reset_vals()
        self.rescale_elos()


if __name__ == '__main__':
    from src.team_trainer import DiscreteInputTrainer
    from src.language_replay_buffer import ReplayBufferDiskStorage
    import os, sys

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))

    torch.random.manual_seed(69)
    popsize = 20

    agents = torch.arange(popsize)%6


    def clone_fn(original, replacements):
        temp = agents.clone()
        agents[original] = temp[replacements]


    class MaxOutcome(OutcomeFn):
        """
        return team with highest indices
        """

        def get_outcome(self, team_choices, train=None):
            agent_choices = [agents[team[0]] for team in team_choices]
            if agent_choices[0] == agent_choices[1]:
                return [(.5, [PlayerInfo()]), (.5, [PlayerInfo()])]

            if agent_choices[0] > agent_choices[1]:
                return [(1, [PlayerInfo()]), (0, [PlayerInfo()])]

            if agent_choices[0] < agent_choices[1]:
                return [(0, [PlayerInfo()]), (1, [PlayerInfo()])]


    cap = CaptianCoevolution(outcome_fn=MaxOutcome(),
                             population_sizes=[popsize],
                             team_trainer=TeamTrainer(num_agents=popsize, ),
                             clone_fn=clone_fn
                             )
    for _ in range(20):
        for _ in range(2):
            cap.epoch(rechoose=False)
        print(cap.captian_elos)
        print(agents)
        cap.epoch(rechoose=True)

    print(cap.captian_elos)
    print(agents)
    quit()

    test = TwoTeamsCaptainCoevolution(population_size=popsize,
                                      outcome_fn=lambda i, j: max(i, j),
                                      clone_fn=clone_fn,
                                      )
    print(agents)
    test.breed()
    print(agents)
