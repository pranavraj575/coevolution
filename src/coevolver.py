import numpy as np
import torch, os, shutil

from src.team_trainer import TeamTrainer
from src.game_outcome import PlayerInfo, OutcomeFn, PettingZooOutcomeFn
from src.zoo_cage import ZooCage
from src.utils.dict_keys import *

from multi_agent_algs.common import DumEnv


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

            TODO: this is mostly unused
            member_to_population: takes team member (team_idx, member_idx) and returns set of
                populations (subset of (0<i<len(population_sizes))) that the member can be drawn from
                by default, assumes each member can be drawn from each population
        """
        self.outcome_fn = outcome_fn
        self.population_sizes = population_sizes
        self.team_sizes = team_sizes
        self.num_teams = len(team_sizes)
        self.env_constructor = lambda: None

        # cumulative population sizes
        self.cumsums = np.cumsum(np.concatenate(([0], self.population_sizes)))

        # total population size
        self.N = int(self.cumsums[-1])

        if self.N%self.num_teams != 0:
            print("WARNING: number of agents is not divisible by num teams")
            print('\tthis is fine, but will have non-uniform game numbers for each agent')

        self.build_pop_to_member_and_team(member_to_population)

    def build_pop_to_member_and_team(self, original_member_to_pop):
        if original_member_to_pop is None:
            original_member_to_pop = lambda team_idx, member_idx: set(range(len(self.population_sizes)))
        self.pop_to_team = [set() for _ in self.population_sizes]
        self.pop_to_member = [set() for _ in self.population_sizes]

        self.team_to_pop = [set() for _ in self.team_sizes]
        self.member_to_pop = [[set() for _ in range(team_size)] for team_size in self.team_sizes]

        self.pop_and_team_to_valid_locations = [
            [torch.zeros(team_size, dtype=torch.bool) for team_size in self.team_sizes]
            for _ in self.population_sizes
        ]
        for team_idx, team_size in enumerate(self.team_sizes):
            for member_idx in range(team_size):
                populations = original_member_to_pop(team_idx, member_idx)
                if populations is None:
                    populations = set(range(len(self.population_sizes)))
                for pop_idx in populations:
                    self.pop_to_member[pop_idx].add((team_idx, member_idx))
                    self.pop_to_team[pop_idx].add(team_idx)

                    self.team_to_pop[team_idx].add(pop_idx)
                    self.member_to_pop[team_idx][member_idx].add(pop_idx)

                    self.pop_and_team_to_valid_locations[pop_idx][team_idx][member_idx] = True

        self.pop_to_team = tuple(self.pop_to_team)
        self.pop_to_member = tuple(self.pop_to_member)

        self.team_to_pop = tuple(self.team_to_pop)
        self.member_to_pop = tuple(tuple(t) for t in self.member_to_pop)

        self.pop_and_team_to_valid_locations = tuple(tuple(t) for t in self.pop_and_team_to_valid_locations)

        pop_members = [set(range(self.cumsums[i], self.cumsums[i + 1]))
                       for i in range(len(self.population_sizes))]
        self.team_to_choices = []
        for team_idx in range(len(self.team_sizes)):
            choices = set()
            for pop_idx in self.team_to_pop[team_idx]:
                choices.update(pop_members[pop_idx])
            self.team_to_choices.append(choices)
        self.team_to_choices = tuple(self.team_to_choices)

        # for each team, is a (team members, self.N) array of which agents are valid choices
        self.team_to_valid_members = []
        for team_idx, team_size in enumerate(self.team_sizes):
            valid_members = torch.zeros((team_size, self.N), dtype=torch.bool)
            for member_idx in range(team_size):
                for pop_idx in self.member_to_pop[team_idx][member_idx]:
                    # every member drawn from this population is valid, so give it a 1
                    valid_members[member_idx, list(pop_members[pop_idx])] = 1
            self.team_to_valid_members.append(valid_members)

    def sample_team_member_from_pop(self, pop_idx):
        for idx in self.pop_to_member[pop_idx]:
            return idx

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

    def create_random_captians(self):
        """
        Returns: iterable of edges (i,j) that is a matching on (0,...,N)
            if N is not divisible by number of teams, a set of agents are chosen twice
            othewise, each agent is chosen once
        """

        unused = set(range(self.N))
        pop_members = [set(range(self.cumsums[i], self.cumsums[i + 1]))
                       for i in range(len(self.population_sizes))]
        # this will always terminate as long as every population is used by at least one team
        # since then the unused set will always drop by at least one each time
        while unused:
            captains = [None for _ in range(self.num_teams)]
            uniques = [None for _ in range(self.num_teams)]
            for team_idx in torch.randperm(self.num_teams):
                choices = self.team_to_choices[team_idx]

                unused_choices = choices.intersection(unused)
                if unused_choices:
                    # if there are unused agents, use them here
                    cap = np.random.choice(list(unused_choices))
                    unique = True
                else:
                    # otherwise, default to using a previous agent
                    cap = np.random.choice(list(choices))
                    unique = False
                captains[team_idx] = cap
                uniques[team_idx] = unique
            yield tuple(captains), tuple(uniques)
            # remove captains from unused
            unused.difference_update(captains)

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
                 team_trainer: TeamTrainer = None,
                 clone_fn=None,
                 team_sizes=(1, 1),
                 elo_conversion=400/np.log(10),
                 elo_update=32*np.log(10)/400,
                 mutation_fn=None,
                 noise_model=None,
                 member_to_population=None,
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
                         member_to_population=member_to_population,
                         )
        if team_trainer is None:
            team_trainer = TeamTrainer(num_agents=self.N)
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
        for team_idx, (captian, unq, team_size) in enumerate(zip(captian_choices, unique, self.team_sizes)):
            captian_pop_idx, _ = self.index_to_pop_index(global_idx=captian)
            valid_locations = self.pop_and_team_to_valid_locations[captian_pop_idx][team_idx]

            team = self.team_trainer.add_member_to_team(member=captian,
                                                        T=team_size,
                                                        N=1,
                                                        noise_model=self.noise_model,
                                                        valid_locations=valid_locations.view((1, team_size)),
                                                        )
            pos = torch.where(team.view(-1) == captian)[0].item()
            captian_positions.append(pos)

            valid_members = self.team_to_valid_members[team_idx]
            team = self.team_trainer.fill_in_teams(initial_teams=team,
                                                   noise_model=self.noise_model,
                                                   valid_members=valid_members.view((1, team_size, self.N)),
                                                   )
            teams.append(team)
            tinfo = []
            for i, member in enumerate(team.flatten()):
                global_idx = member.item()
                captian_pop_idx, local_idx = self.index_to_pop_index(global_idx=global_idx)

                info = self.get_info(pop_idx=captian_pop_idx, local_idx=local_idx)
                if i == pos:
                    info[TEMP_DICT_CAPTIAN] = True
                    info[TEMP_DICT_UNIQUE] = unq
                else:
                    info[TEMP_DICT_CAPTIAN] = False
                tinfo.append(info)
            train_infos.append(tinfo)
        team_outcomes = self.outcome_fn.get_outcome(team_choices=teams,
                                                    train_infos=train_infos,
                                                    env=self.env_constructor(),
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
                 zoo_dir,
                 worker_constructors,
                 member_to_population=None,
                 team_trainer: TeamTrainer = None,
                 team_sizes=(1, 1),
                 elo_conversion=400/np.log(10),
                 elo_update=32*np.log(10)/400,
                 noise_model=None,
                 reinit_agents=True,
                 mutation_prob=.01,
                 protect_new=20,
                 team_idx_to_agent_id=None,
                 worker_constructors_from_env_input=False,
                 ):
        """
        Args:
            env_constructor:
            outcome_fn:
            population_sizes: list of K ints, size of each population
            team_trainer:
            worker_constructors: if specified, list of K functions, size of population
                each funciton goes from (index, enviornment) into
                    (a worker with the specified action and obs space, worker_info) tuple
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
                         member_to_population=member_to_population,
                         )

        self.env_constructor = env_constructor

        test_env = self.env_constructor()

        if team_idx_to_agent_id is not None:
            self.team_idx_to_agent_id = team_idx_to_agent_id
        else:
            env_agents = iter(test_env.agents)
            dict_team_idx_to_agent_id = dict()
            for team_idx, team_size in enumerate(self.team_sizes):
                for member_idx in range(team_size):
                    dict_team_idx_to_agent_id[(team_idx, member_idx)] = next(env_agents)
            self.team_idx_to_agent_id = lambda idx: dict_team_idx_to_agent_id[idx]

        self.action_space = test_env.action_space
        self.observation_space = test_env.observation_space
        if worker_constructors_from_env_input:
            def pop_idx_to_dumenv(pop_idx):
                idx = self.sample_team_member_from_pop(pop_idx=pop_idx)
                agent_id = self.team_idx_to_agent_id(idx=idx)
                return DumEnv(action_space=self.action_space(agent_id),
                              obs_space=self.observation_space(agent_id),
                              )

            self.worker_constructors = lambda pop_idx: (lambda i:
                                                        worker_constructors[pop_idx](i,
                                                                                     env=pop_idx_to_dumenv(pop_idx)
                                                                                     )
                                                        )

        else:
            self.worker_constructors = lambda pos_idx: worker_constructors[pos_idx]
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
        agent, info = self.worker_constructors(pop_idx)(local_idx)
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

        def get_outcome(self, team_choices, train_infos=None, env=None):
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

    from multi_agent_algs.dqn.DQN import WorkerDQN
    from stable_baselines3.dqn import MlpPolicy
    from pettingzoo.classic import tictactoe_v3

    capzoo = PettingZooCaptianCoevolution(env_constructor=lambda : None,
                                          outcome_fn=MaxOutcome(),
                                          population_sizes=[3, 4, 5],
                                          team_trainer=TeamTrainer(num_agents=3 + 4 + 5),
                                          worker_constructors=[lambda *_: (WorkerDQN(policy=MlpPolicy,
                                                                                     env='CartPole-v1'), None)
                                                               for _ in range(3)],
                                          zoo_dir=os.path.join(DIR, 'data', 'coevolver_zoo_test'),
                                          )
    capzoo.kill_zoo()
