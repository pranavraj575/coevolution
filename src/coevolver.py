import shutil

import numpy as np
import torch, os
from src.team_trainer import TeamTrainer
from src.game_outcome import PlayerInfo, OutcomeFn, PettingZooOutcomeFn
from src.zoo_cage import ZooCage, DICT_IS_WORKER
from src.utils.dict_keys import *


class CoevolutionBase:
    """
    general coevolution algorithm
    """

    def __init__(self,
                 outcome_fn: OutcomeFn,
                 population_sizes,
                 team_sizes=(1, 1),
                 member_to_population=None,
                 ):
        """
        Args:
            outcome_fn: collect outcomes and do RL training
            population_sizes: list of number of agents in each population
                usually can just be a list of one element
                multiple populations are useful if there are different 'types' of agents
                    in the game that take different inputs/have access to different actions
            team_sizes: tuple of number of agents in each team
                i.e. (1,1) is a 1v1 game

            TODO: this is currently unused
            member_to_population: takes team member (team_idx, member_idx) and returns set of
                populations (subset of (0<i<len(population_sizes))) that the member can be drawn from
                by default, assumes each member can be drawn from each population
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
        if member_to_population is None:
            self.member_to_population = lambda: set(range(len(self.population_sizes)))
        else:
            self.member_to_population = member_to_population
        if member_to_population is None:

            self.pop_to_team = lambda: set(range(len(team_sizes)))
        else:
            self.build_pop_to_team()

    def build_pop_to_team(self):
        temp = [set() for _ in range(len(self.population_sizes))]
        for team_idx, team_size in enumerate(self.team_sizes):
            for member_idx in range(team_size):
                populations = self.member_to_population(team_idx, member_idx)
                if populations is None:
                    for item in temp:
                        item.add(team_idx)
                else:
                    for pop_idx in populations:
                        temp[pop_idx].add(team_idx)
        temp = tuple(tuple(item) for item in temp)
        self.pop_to_team = lambda pop_idx: temp[pop_idx]

    def index_to_pop_index(self, global_idx):
        """
        returns the index of agent specified by i
        Args:
            global_idx:
        Returns:
            population index (which population), index in population
        """
        pop_idx = 0
        while global_idx - self.population_sizes[pop_idx] >= 0:
            global_idx -= self.population_sizes[pop_idx]
            pop_idx += 1
        return pop_idx, global_idx

    def pop_index_to_index(self, pop_idx, local_idx):
        return self.cumsums[pop_idx] + local_idx

    def get_info(self, pop_idx, local_idx):
        return dict()

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
            tinfo = []
            for i, member in enumerate(team.flatten()):
                global_idx = member.item()
                pop_idx, local_idx = self.index_to_pop_index(global_idx=global_idx)

                info = self.get_info(pop_idx=pop_idx, local_idx=local_idx)
                if i == pos:
                    info[TEMP_DICT_CAPTIAN] = True
                    info[TEMP_DICT_UNIQUE] = unq
                else:
                    info[TEMP_DICT_CAPTIAN] = False
                tinfo.append(info)
            train_infos.append(tinfo)
        team_outcomes = self.outcome_fn.get_outcome(team_choices=teams,
                                                    train_infos=train_infos,
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
                 zoo_dir,
                 team_sizes=(1, 1),
                 elo_conversion=400/np.log(10),
                 elo_update=32*np.log(10)/400,
                 noise_model=None,
                 reinit_agents=True,
                 mutation_prob=.01,
                 protect_new=20,
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
                info dicts keys are found in src.utils.dict_keys

                worker_info relevant keys:
                    DICT_IS_WORKER: whether agent is a worker class
                    DICT_CLONABLE: whether agent is able to be cloned
                    DICT_CLONE_REPLACABLE: whether agent is able to be replace by a clone of another agent
                    DICT_MUTATION_REPLACABLE: whether agent is resettable in mutation
                    DICT_KEEP_OLD_BUFFER: whether to keep old buffer in event of reset
                    DICT_POSITION_DEPENDENT: set of dict that is associated with position:
                        i.e. if any entries are in this set, they will be unchanged after replacing agent with a clone
                            or mutation
            zoo_dir: place to store cages of agents
            team_sizes:
            elo_conversion:
            elo_update:
            noise_model:
            mutation_prob: probability an agent randomly reinitializes each epoch
            protect_new: protect agents younger than this
        """
        super().__init__(outcome_fn=outcome_fn,
                         population_sizes=population_sizes,
                         team_trainer=team_trainer,
                         team_sizes=team_sizes,
                         elo_conversion=elo_conversion,
                         elo_update=elo_update,
                         mutation_fn=None,
                         clone_fn=None,
                         noise_model=noise_model,
                         )

        self.env_constructor = env_constructor
        self.worker_constructors = worker_constructors
        self.zoo = [
            ZooCage(zoo_dir=os.path.join(zoo_dir, 'cage_' + str(i)),
                    overwrite_zoo=True,
                    )
            for i in range(len(self.population_sizes))
        ]

        self.outcome_fn: PettingZooOutcomeFn
        self.outcome_fn.set_zoo(self.zoo)
        self.outcome_fn.set_index_conversion(index_conversion=self.index_to_pop_index)
        self.zoo_dir = zoo_dir
        if reinit_agents:
            self.init_agents()
        self.mutation_prob = mutation_prob
        self.protect_new = protect_new

    def init_agents(self):
        for pop_idx, popsize in enumerate(self.population_sizes):
            for i in range(popsize):
                self.reset_agent(pop_idx=pop_idx, local_idx=i)

    def save_zoo(self, save_dir):
        """
        saves all zoo cages to specified dir
        Args:
            save_dir: dir to save to
        """
        for zoo_cage in self.zoo:
            zoo_cage.save_cage(os.path.join(save_dir, os.path.basename(zoo_cage.zoo_dir)))

    def kill_zoo(self):
        for zoo_cage in self.zoo:
            zoo_cage.kill_cage()
        shutil.rmtree(self.zoo_dir)

    def get_info(self, pop_idx, local_idx):
        return self.zoo[pop_idx].load_info(key=str(local_idx))

    def breed(self):
        valid_replacement_indices = []
        elos = []
        for global_idx in range(self.N):
            pop_idx, local_idx = self.index_to_pop_index(global_idx=global_idx)
            info = self.get_info(pop_idx=pop_idx, local_idx=local_idx)
            if info.get(DICT_CLONABLE, True):
                valid_replacement_indices.append(global_idx)
                elos.append(self.captian_elos[global_idx])
        if not valid_replacement_indices:
            # no clonable agents
            return

        dist = torch.softmax(torch.tensor(elos), dim=-1)
        # sample from this distribution with replacements
        replacements = list(torch.multinomial(dist, self.N, replacement=True))
        for global_idx, rep_idx in enumerate(replacements):
            # replace agent at global idx with agent at valid_replacement_indices[rep_idx]
            pop_idx, local_idx = self.index_to_pop_index(global_idx=global_idx)
            info = self.get_info(pop_idx=pop_idx, local_idx=local_idx)
            # only replace old agents

            if info.get(DICT_CLONE_REPLACABLE, True) and (info.get(DICT_AGE, 0) > self.protect_new):
                # keep these values,
                position_dependent_keys = info.get(DICT_POSITION_DEPENDENT, set())
                position_dependent_keys.add(DICT_POSITION_DEPENDENT)

                pop_idx_rep, i_rep = self.index_to_pop_index(global_idx=valid_replacement_indices[rep_idx])
                replacement_agent, replacement_info = self.zoo[pop_idx_rep].load_animal(key=str(i_rep),
                                                                                        load_buffer=True)
                replacement_info[DICT_AGE] = 0
                # update replacement dictionary with elements in position_dependent_keys dict
                for key in position_dependent_keys:
                    if key in info:
                        # copy the key over
                        replacement_info[key] = info[key]
                    elif key in replacement_info:
                        # otherwise if key is in replacement info, remove it
                        replacement_info.pop(key)

                self.zoo[pop_idx].overwrite_animal(animal=replacement_agent,
                                                   key=str(local_idx),
                                                   info=replacement_info,
                                                   save_buffer=True,
                                                   save_class=True,
                                                   )
                # replace elo as well
                self.captian_elos[global_idx] = elos[rep_idx]
        self.rebase_elos()
        self.age_up_all_agents()

    def age_up_all_agents(self):
        for pop_idx, popsize in enumerate(self.population_sizes):
            for local_idx in range(popsize):
                self.age_up_agent(pop_idx=pop_idx, local_idx=local_idx)

    def age_up_agent(self, pop_idx, local_idx):
        info = self.get_info(pop_idx=pop_idx, local_idx=local_idx)
        info[DICT_AGE] = info.get(DICT_AGE, 0) + 1
        self.zoo[pop_idx].save_info(key=str(local_idx), info=info)

    def reset_agent(self, pop_idx, local_idx, keep_old_buffer=False):
        agent, info = self.worker_constructors[pop_idx](local_idx)
        # info[PERSONAL_DICT_AGE] = 0
        cage = self.zoo[pop_idx]
        if info.get(DICT_IS_WORKER, True):
            if keep_old_buffer and cage.worker_exists(worker_key=str(local_idx)):
                old_worker, _ = cage.load_worker(worker_key=str(local_idx),
                                                 WorkerClass=None,
                                                 load_buffer=True,
                                                 )
            else:
                old_worker = None
            cage.overwrite_worker(worker=agent,
                                  worker_key=str(local_idx),
                                  save_buffer=True,
                                  save_class=True,
                                  worker_info=info,
                                  )
            if old_worker is not None:
                cage.update_worker_buffer(local_worker=old_worker,
                                          worker_key=str(local_idx),
                                          WorkerClass=None,
                                          )
        else:
            cage.overwrite_other(other=agent,
                                 other_key=str(local_idx),
                                 other_info=info,
                                 )
        # reset elo to average
        self.captian_elos[self.pop_index_to_index(pop_idx=pop_idx, local_idx=local_idx)] = torch.mean(self.captian_elos)

    def mutate(self):
        if self.mutation_prob > 0:
            for global_idx in range(self.N):
                if torch.rand(1) < self.mutation_prob:
                    pop_idx, local_idx = self.index_to_pop_index(global_idx=global_idx)
                    info = self.get_info(pop_idx=pop_idx, local_idx=local_idx)
                    if info.get(DICT_MUTATION_REPLACABLE, True) and (info.get(DICT_AGE, 0) > self.protect_new):
                        self.reset_agent(pop_idx=pop_idx,
                                         local_idx=local_idx,
                                         keep_old_buffer=info.get(DICT_KEEP_OLD_BUFFER, True)
                                         )


if __name__ == '__main__':
    import os, sys

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))

    torch.random.manual_seed(69)
    popsize = 20

    agents = torch.arange(popsize)%6


    def clone_fn(original, replacements):
        temp = agents.clone()
        agents[original] = temp[replacements]


    class MaxOutcome(PettingZooOutcomeFn):
        """
        return team with highest indices
        """

        def get_outcome(self, team_choices, train_info=None):
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

    from parallel_algs.dqn.DQN import WorkerDQN
    from stable_baselines3.dqn import MlpPolicy

    capzoo = PettingZooCaptianCoevolution(env_constructor=lambda _: None,
                                          outcome_fn=MaxOutcome(),
                                          population_sizes=[3, 4, 5],
                                          team_trainer=TeamTrainer(num_agents=3 + 4 + 5),
                                          worker_constructors=[lambda _: (WorkerDQN(policy=MlpPolicy,
                                                                                    env='CartPole-v1'), None)
                                                               for _ in range(3)],
                                          zoo_dir=os.path.join(DIR, 'data', 'coevolver_zoo_test'),
                                          )
    capzoo.kill_zoo()
