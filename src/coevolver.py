import numpy as np
import torch
from src.team_trainer import TeamTrainer
from src.game_outcome import PlayerInfo, OutcomeFn, PettingZooOutcomeFn
from src.zoo_cage import ZooCage


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

        # cumulative population sizes
        self.cumsums = np.cumsum(np.concatenate(([0], self.population_sizes)))

        # total population size
        self.N = int(self.cumsums[-1])

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

    def pop_index_to_index(self, pop_idx, j):
        return self.cumsums[pop_idx] + j

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
            yield tuple(choice), tuple(True for _ in choice)
        if unused:
            remaining = self.num_teams - len(unused)
            unique = tuple(True for _ in range(len(unused))) + tuple(False for _ in range(remaining))

            # randomly sort unused, and concatenate with a random choice of the other agent indices
            choice = (tuple(np.random.choice(list(unused), len(unused), replace=False)) +
                      tuple(np.random.choice(list(set(range(N)).difference(unused)), remaining, replace=False)))
            yield choice, unique

    def epoch(self, rechoose=True):
        # TODO: parallelizable
        for choice, unique in self.create_random_captians():
            self.train_and_update_results(captian_choices=choice, unique=unique)
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

    def train_and_update_results(self, captian_choices, unique):
        """
        takes a choice of team captians and trains them in RL environment
        updates variables to reflect result of game(s)
        Args:
            captian_choices: tuple of self.num_teams indices
            unique: whether each captian is unique (each captian will be marked unique exactly once
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

    def train_and_update_results(self, captian_choices, unique):
        """
        takes a choice of team captians and trains them in RL environment
        updates variables to reflect result of game(s)
        Args:
            captian_choices: tuple of self.num_teams indices
            unique: whether each captian is unique (each captian will be marked unique exactly once
        """
        teams = []
        train_infos = []
        captian_positions = []
        # expected win probabilities, assuming the teams win probability is determined by captian elo
        expected_win_probs = torch.softmax(self.captian_elos[captian_choices,],
                                           dim=-1)
        for captian, unq, team_size in zip(captian_choices, unique, self.team_sizes):
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
            tinfo = [{'captian': False} for _ in range(team_size)]
            tinfo[pos] = {
                'captian': True,
                'unique': unq
            }
            train_infos.append(tinfo)
        team_outcomes = self.outcome_fn.get_outcome(team_choices=teams,
                                                    train_info=train_infos,
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
                 outcome_fn: PettingZooOutcomeFn,
                 population_sizes,
                 team_trainer: TeamTrainer,
                 worker_constructors,
                 zoo_cages_dir,
                 team_sizes=(1, 1),
                 elo_conversion=400/np.log(10),
                 elo_update=32*np.log(10)/400,
                 noise_model=None,
                 reinit_agents=True,
                 ):
        """
        Args:
            env_constructor:
            outcome_fn:
            population_sizes: list of K ints, size of each population
            team_trainer:
            worker_constructors: list of K constructors, size of each population
                ith constructor takes a worker index (0<j<population_sizes[i])
                    returns a (worker,worker_info) tuple
            zoo_cages_dir: place to store cages of agents
            team_sizes:
            elo_conversion:
            elo_update:
            noise_model:
        """
        super().__init__(outcome_fn=outcome_fn,
                         population_sizes=population_sizes,
                         team_trainer=team_trainer,
                         team_sizes=team_sizes,
                         elo_conversion=elo_conversion,
                         elo_update=elo_update,
                         mutation_fn=None,
                         clone_fn=None,
                         noise_model=noise_model
                         )
        self.env_constructor = env_constructor
        self.worker_constructors = worker_constructors
        self.zoo = [
            ZooCage(zoo_dir=os.path.join(zoo_cages_dir, 'cage_' + str(i)),
                    overwrite_zoo=True,
                    )
            for i in range(self.num_teams)
        ]

        self.outcome_fn: PettingZooOutcomeFn
        self.outcome_fn.set_zoo(self.zoo)

        if reinit_agents:
            self.init_agents()

    def init_agents(self):
        for popsize, constructor, cage in zip(self.population_sizes,
                                              self.worker_constructors,
                                              self.zoo,
                                              ):
            for i in range(popsize):
                worker, worker_info = constructor(i)
                cage.overwrite_worker(worker=worker,
                                      worker_key=str(i),
                                      save_buffer=True,
                                      save_class=True,
                                      worker_info=worker_info,
                                      )

    def save_zoo(self, save_dir):
        """
        saves all zoo cages to specified dir
        Args:
            save_dir: dir to save to
        """
        for zoo_cage in self.zoo:
            zoo_cage.save_cage(os.path.join(save_dir, os.path.basename(zoo_cage.zoo_dir)))

    def breed(self):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError


if __name__ == '__main__':
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
