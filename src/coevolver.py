import numpy as np
import torch, os, shutil, pickle

from src.team_trainer import TeamTrainer
from src.game_outcome import PlayerInfo, OutcomeFn, PettingZooOutcomeFn
from src.zoo_cage import ZooCage
from src.utils.dict_keys import (DICT_AGE,
                                 DICT_CLONABLE,
                                 DICT_MUTATION_REPLACABLE,
                                 DICT_CLONE_REPLACABLE,
                                 DICT_POSITION_DEPENDENT,
                                 DICT_KEEP_OLD_BUFFER,
                                 DICT_UPDATE_WITH_OLD_BUFFER,
                                 TEMP_DICT_CAPTIAN,
                                 TEMP_DICT_CAPTIAN_UNIQUE,
                                 COEVOLUTION_DICT_CAPTIAN_ELO,
                                 COEVOLUTION_DICT_ELO_UPDATE,
                                 COEVOLUTION_DICT_ELO_CONVERSION,
                                 )

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

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
        self.original_member_to_population = member_to_population
        self.info = {'epochs': 0,
                     'epoch_infos': [],
                     }

    def clear(self):
        """
        clears any disc storage
        """
        pass

    def save(self, save_dir):

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        f = open(os.path.join(save_dir, 'info.pkl'), 'wb')
        pickle.dump(self.info, f)
        f.close()

    def load(self, save_dir):
        f = open(os.path.join(save_dir, 'info.pkl'), 'rb')
        self.info.update(pickle.load(f))
        f.close()

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

    def pop_index_to_index(self, pop_local_idx):
        pop_idx, local_idx = pop_local_idx
        return self.cumsums[pop_idx] + local_idx

    def get_info(self, pop_local_idx):
        return dict()

    def create_random_captians(self):
        """
        Returns: iterable of edges (i,j) that is a matching on (0,...,N)
            if N is not divisible by number of teams, a set of agents are chosen twice
            othewise, each agent is chosen once
        """

        unused = set(range(self.N))
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

    def epoch(self, rechoose=True, save_epoch_info=True):
        # TODO: parallelizable
        epoch_info = {
            'epoch': self.info['epochs'],
            'episodes': []
        }
        for choice, unique in self.create_random_captians():
            episode_info = self.train_and_update_results(captian_choices=choice, unique=unique)
            epoch_info['episodes'].append(episode_info)
        if rechoose:
            epoch_info['breeding'] = self.breed()
            epoch_info['mutation'] = self.mutate()
        self.info['epochs'] += 1
        if save_epoch_info:
            self.info['epoch_infos'].append(epoch_info)

    def breed(self):
        """
        terminates poorly performing agents and replaces them with clones
            can also replace through sexual reproduction, if that is possible

        also updates internal variables to reflect the new agents
        Returns:
            breeding info
        """
        raise NotImplementedError

    def mutate(self):
        """
        mutates/reinitializes agents at random
        Returns:
            mutation info
        """
        pass

    def train_and_update_results(self, captian_choices, unique):
        """
        takes a choice of team captians and trains them in RL environment
        updates variables to reflect result of game(s)
        Args:
            captian_choices: tuple of self.num_teams indices
            unique: whether each captian is unique (each captian will be marked unique exactly once
        Returns: info
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
        self.mutation_fn = mutation_fn
        self.info.update({
            COEVOLUTION_DICT_CAPTIAN_ELO: torch.zeros(self.N),
            COEVOLUTION_DICT_ELO_UPDATE: elo_update,
            COEVOLUTION_DICT_ELO_CONVERSION: elo_conversion,
        })

    def clear(self):
        super().clear()
        self.team_trainer.clear()

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        self.team_trainer.save(save_dir=os.path.join(save_dir, 'team_trainer'))

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        self.team_trainer.load(save_dir=os.path.join(save_dir, 'team_trainer'))

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
        episode_info = {'captian_choices': captian_choices,
                        'unique_captians': unique
                        }
        teams = []
        train_infos = []
        captian_positions = []
        # expected win probabilities, assuming the teams win probability is determined by captian elo
        expected_win_probs = torch.softmax(self.captian_elos[captian_choices,],
                                           dim=-1)
        # team selection (pretty much does nothing if teams are size 1
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

                info = self.get_info(pop_local_idx=(captian_pop_idx, local_idx))
                if i == pos:
                    info[TEMP_DICT_CAPTIAN] = True
                    info[TEMP_DICT_CAPTIAN_UNIQUE] = unq
                else:
                    info[TEMP_DICT_CAPTIAN] = False
                tinfo.append(info)
            train_infos.append(tinfo)
        episode_info['teams'] = tuple(team.detach().numpy() for team in teams)
        team_outcomes = self.outcome_fn.get_outcome(team_choices=teams,
                                                    train_infos=train_infos,
                                                    env=self.env_constructor(),
                                                    )
        episode_info['team_outcomes'] = tuple(t for t, _ in team_outcomes)
        for (captian,
             team,
             (team_outcome, player_infos),
             expected_outcome) in zip(captian_choices,
                                      teams,
                                      team_outcomes,
                                      expected_win_probs):
            self.captian_elos[captian] += self.elo_update*(team_outcome - expected_outcome)

            # combine all player observations into one (can also just add each individual PlayerInfo)
            combined_obs = PlayerInfo()
            for player_info in player_infos:
                combined_obs.union_obs(other_player_info=player_info)

            self.team_trainer.add_to_buffer(scalar=team_outcome,
                                            obs_preembed=combined_obs.obs_preembed,
                                            team=team,
                                            obs_mask=combined_obs.obs_mask,
                                            )
        return episode_info

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
        return

    def mutate(self):
        if self.mutation_fn is not None:
            return self.mutation_fn()
        else:
            return super().mutate()

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
        self.set_captian_elos(self._get_rebased_elos(
            elos=self.captian_elos,
            base_elo=base_elo,
        )
        )

    def get_classic_elo(self, base_elo=None):
        """
        gets 'classic' elo values
        Args:
            base_elo: if specified, rebases the elos so this is the average elo
        Returns:
            'classic' elos, such that for players with 'classic' elos Ra', Rb', the win probability of a is
                1/(1+e^{(Rb'-Ra')/self.info['elo_conversion']})
            i.e. if the default elo_conversion is used, this is  1/(1+10^{(Rb'-Ra')/400}), the standard value
        """

        scaled_elos = self.captian_elos*self.elo_conversion
        if base_elo is not None:
            scaled_elos = self._get_rebased_elos(elos=scaled_elos, base_elo=base_elo)
        return scaled_elos

    def get_inverted_distribution(self, elos):
        """
        gets distribution according to inverted elos (i.e. more likely to pick bad agents)
            currently picks according to softmax(-elos)
                this has the effect of choosing each agent proportional to 1/(win probability)

            another method would be to pick proportional to 1-softmax(elos)
                however, this method does not differentiate well between 'bad' and 'very bad' agents
        Args:
            elos: elos to convert into a distribution
        Returns:
            normalized distribution
        """
        return torch.softmax(-torch.tensor(elos), dim=-1)

    def set_captian_elos(self, captian_elos):
        self.info[COEVOLUTION_DICT_CAPTIAN_ELO] = captian_elos

    @property
    def captian_elos(self):
        # note: since this returns the reference to a tensor
        # this can only be used to mutate indices
        # i.e. self.captian_elos[2]=3 mutates self.captian_elos
        # self.captian_elos=torch.rand(2) does not mutate captian_elos
        return self.info[COEVOLUTION_DICT_CAPTIAN_ELO]

    @property
    def elo_update(self):
        return self.info[COEVOLUTION_DICT_ELO_UPDATE]

    @property
    def elo_conversion(self):
        return self.info[COEVOLUTION_DICT_ELO_CONVERSION]


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
        test_env.reset()

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
        # self.worker_constructors is now (pos_idx -> (local idx -> agent))

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

    def clear(self):
        super().clear()
        self.clear_zoo()

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        self.save_zoo(save_dir=os.path.join(save_dir, 'zoo'))

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        self.load_zoo(save_dir=os.path.join(save_dir, 'zoo'))

    def init_agents(self):
        for pop_idx, popsize in enumerate(self.population_sizes):
            for local_idx in range(popsize):
                self.reset_agent(pop_local_idx=(pop_idx, local_idx),
                                 elo=torch.mean(self.captian_elos),
                                 )

    def reset_agent(self, pop_local_idx, elo=None):
        pop_idx, local_idx = pop_local_idx
        agent, info = self.worker_constructors(pop_idx)(local_idx)
        cage = self.zoo[pop_idx]
        cage.overwrite_animal(animal=agent,
                              key=str(local_idx),
                              info=info,
                              save_buffer=True,
                              save_class=True,
                              )
        if elo is not None:
            self.captian_elos[self.pop_index_to_index(pop_local_idx=pop_local_idx)] = elo

    def get_info(self, pop_local_idx):
        pop_idx, local_idx = pop_local_idx
        return self.zoo[pop_idx].load_info(key=str(local_idx))

    def breed(self):
        return self.classic_breed()

    def _get_valid_idxs(self, validity_fn):
        """
        returns indexes of all 'valid agents' according to check
        Args:
            validity_fn: (info dict -> whether to return agent)
        Returns:
            iterable of global indexes of replacable agents
        """

        for global_idx in range(self.N):
            pop_local_idx = self.index_to_pop_index(global_idx=global_idx)
            info = self.get_info(pop_local_idx=pop_local_idx)
            if validity_fn(info):
                yield global_idx

    def conservative_breed(self, number_to_replace, base_elo=0., force_replacements=True):
        """
        Args:
            number_to_replace: max number of agents to try to replace (must be in range [0,self.N])
            base_elo: if not None, rebases all elos so this value is average
            force_replacements: if True, always replaces number_to_replace
                (unless there are fewer potential target agents than number_to_replace)
        keeps rest of the agents constant
            selects agents according to softmax of inverted ELO
                i.e. takes negative of the ELO and uses that to select 'bad' agents to replace
            tries to replace this agent, (checks DICT_AGE and DICT_IS_CLONE_REPLACABLE)

            chooses replacements with softmax of standard ELO
        """
        breed_dic = {'number_replaced': 0}
        if number_to_replace <= 0:
            return breed_dic

        # pick the agents to potentially clone
        candidate_clone_idxs = list(self._get_valid_idxs(validity_fn=
                                                         lambda info:
                                                         info.get(DICT_CLONABLE, True)
                                                         )
                                    )
        if not candidate_clone_idxs:
            # no clonable agents
            return breed_dic

        # pick the agents to potentially replace with a clone
        if force_replacements:
            candidate_target_idxs = list(
                self._get_valid_idxs(validity_fn=lambda info: info.get(DICT_CLONE_REPLACABLE, True) and
                                                              (info.get(DICT_AGE, 0) > self.protect_new)
                                     ))
        else:
            candidate_target_idxs = list(range(self.N))
        # can replace at most this number
        number_to_replace = min(number_to_replace, len(candidate_target_idxs))
        if not candidate_target_idxs:
            return breed_dic
        # distribution of agents based on how bad they are
        candidate_target_dist = self.get_inverted_distribution(elos=self.captian_elos[candidate_target_idxs])
        # pick a random subset of target agents to replace based on this distribution
        target_idx_idxs = torch.multinomial(candidate_target_dist, number_to_replace, replacement=False)
        # these are the global indexes of the targets
        target_global_idxs = [candidate_target_idxs[target_idx_idx] for target_idx_idx in target_idx_idxs]
        target_elos = [self.captian_elos[target_global_idx] for target_global_idx in target_global_idxs]

        # now pick which agents to clone based on elo
        candidate_clone_elos = self.captian_elos[candidate_clone_idxs]
        clone_dist = torch.softmax(candidate_clone_elos, dim=-1)
        # sample from this distribution with replacement
        clone_idx_idxs = list(torch.multinomial(clone_dist, len(target_global_idxs), replacement=True))
        # element clone_idx_idx in clone_idx_idxs denotes that candidate_clone_idxs[clone_idx_idx] should be cloned
        # also candidate_clone_idxs[clone_idx_idx] has elo clone_elos[clone_idx_idx]

        # global indexes of clones, as well as elos of the clones
        clone_global_idxs = [candidate_clone_idxs[clone_idx_idx] for clone_idx_idx in clone_idx_idxs]
        clone_elos = [candidate_clone_elos[clone_idx_idx] for clone_idx_idx in clone_idx_idxs]

        breed_dic['target_agents'] = []
        breed_dic['target_elos'] = []
        breed_dic['cloned_agents'] = []
        breed_dic['cloned_elos'] = []

        for target_global_idx, target_elo, clone_global_idx, clone_elo in zip(target_global_idxs,
                                                                              target_elos,
                                                                              clone_global_idxs,
                                                                              clone_elos):
            target_info = self.get_info(pop_local_idx=self.index_to_pop_index(global_idx=target_global_idx))
            # check if target is actually replacable by a clone
            if target_info.get(DICT_CLONE_REPLACABLE, True) and (target_info.get(DICT_AGE, 0) > self.protect_new):
                # in that case, replace target with clone
                clone_pop_local_idx = self.index_to_pop_index(global_idx=clone_global_idx)
                clone_agent, clone_info = self.load_animal(pop_local_idx=clone_pop_local_idx, load_buffer=True)
                self.replace_agent(pop_local_idx=self.index_to_pop_index(global_idx=target_global_idx),
                                   replacement=(clone_agent, clone_info),
                                   elo=clone_elo,
                                   keep_old_buff=clone_info.get(DICT_KEEP_OLD_BUFFER, False),
                                   update_with_old_buff=clone_info.get(DICT_UPDATE_WITH_OLD_BUFFER, True),
                                   )
                breed_dic['number_replaced'] += 1
                breed_dic['target_agents'].append(target_global_idx)
                breed_dic['cloned_agents'].append(clone_global_idx)
                breed_dic['cloned_elos'].append(clone_elo)
                breed_dic['target_elos'].append(target_elo)
                # note: it is probably necessary to save clone_elo and target_elo lists beforehand as self.captain_elos
                # are being reassigned with self.replace_agent

        breed_dic['based_elos'] = base_elo
        if base_elo is not None:
            self.rebase_elos(base_elo=base_elo)
        self.age_up_all_agents()
        return breed_dic

    def load_animal(self, pop_local_idx, load_buffer=True):
        pop_idx, local_idx = pop_local_idx
        return self.zoo[pop_idx].load_animal(key=str(local_idx),
                                             load_buffer=load_buffer,
                                             )

    def classic_breed(self):
        """
        breeds according to softmax selection
        https://ieeexplore.ieee.org/document/9308290
        """
        # equivalent to conservative_breed with targets being all agents
        return self.conservative_breed(number_to_replace=self.N)

    def mutate(self):
        if self.mutation_prob == 0:
            return {'num_mutated': 0}
        mutation_dict = dict()

        mutatable_idxs = list(
            self._get_valid_idxs(validity_fn=lambda info: info.get(DICT_MUTATION_REPLACABLE, True) and
                                                          (info.get(DICT_AGE, 0) > self.protect_new)
                                 ))
        # get the number to mutate, choose each with probability self.mutation_prob
        # equivalent to the sum of a bunch of bernoulli variables
        num_to_mutate = torch.sum(torch.bernoulli(
            torch.tensor([self.mutation_prob for _ in mutatable_idxs]
                         )
        )).item()
        mutation_dict['num_mutated'] = num_to_mutate
        if num_to_mutate > 0:
            # distribution based on how bad each agent is
            mut_dist = self.get_inverted_distribution(elos=self.captian_elos[mutatable_idxs])
            # chose which to mutate without replacement
            mut_idx_idxs = torch.multinomial(mut_dist, num_to_mutate, replacement=False)
            # mutatable_idxs[mut_idx_idx] is the global index of an agent to mutate
            mut_global_idxs = [mutatable_idxs[mut_idx_idx] for mut_idx_idx in mut_idx_idxs]
            mutation_dict['idxs_mutated'] = mut_global_idxs

            elo_replacement = torch.mean(self.captian_elos).item()
            mutation_dict['elo_replacement'] = elo_replacement
            for mut_global_idx in mut_global_idxs:
                mut_pop_local_idx = self.index_to_pop_index(global_idx=mut_global_idx)

                info = self.get_info(pop_local_idx=mut_pop_local_idx)
                mut_pop_idx, mut_local_idx = mut_pop_local_idx
                self.replace_agent(pop_local_idx=mut_pop_local_idx,
                                   replacement=self.worker_constructors(mut_pop_idx)(mut_local_idx),
                                   keep_old_buff=info.get(DICT_KEEP_OLD_BUFFER, False),
                                   update_with_old_buff=info.get(DICT_UPDATE_WITH_OLD_BUFFER, True),
                                   elo=elo_replacement,
                                   )
        return mutation_dict

    def replace_agent(self,
                      pop_local_idx,
                      replacement,
                      keep_old_buff=False,
                      update_with_old_buff=True,
                      elo=None,
                      ):
        """
        replaces agent at specified idx with replacement agent
        Args:
            pop_local_idx: pop_idx,local_idx of agent to replace
            replacement: (agent, agent info) to replace agent with
                note that specified entries of info of old agent are carried over
                    specifically, the entries in info[DICT_POSITION_DEPENDENT]
            keep_old_buff: whether to just keep the buffer of old agent
                only works if both both agents share a buffer type
                    (i.e. both OnPolicyAlgorithm or both OffPolicyAlgorithm)
            update_with_old_buff: whether to update replacement agent's buffer with old agent
                only works if both both agents share a buffer type
                    (i.e. both OnPolicyAlgorithm or both OffPolicyAlgorithm)
            elo: elo to set new agent to
        """
        replacement_agent, replacement_info = replacement
        old_worker = None
        (pop_idx, local_idx) = pop_local_idx
        if keep_old_buff or update_with_old_buff:
            old_worker, _ = self.load_animal(pop_local_idx=pop_local_idx, load_buffer=True)
            # if the types of the old worker and replacement worker buffers are not the same, we cannot load old buffer
            if not (
                    (isinstance(old_worker, OnPolicyAlgorithm) and
                     isinstance(replacement_agent, OnPolicyAlgorithm) and
                     type(replacement_agent.rollout_buffer) == type(old_worker.rollout_buffer)
                    ) or
                    (isinstance(old_worker, OffPolicyAlgorithm) and
                     isinstance(replacement_agent, OffPolicyAlgorithm) and
                     type(replacement_agent.replay_buffer) == type(old_worker.replay_buffer)
                    )
            ):
                old_worker = None

        info = self.get_info(pop_local_idx=(pop_idx, local_idx))
        # keep these values
        position_dependent_keys = info.get(DICT_POSITION_DEPENDENT,
                                           # by default we want to keep replacability values
                                           {
                                               DICT_CLONE_REPLACABLE,
                                               DICT_MUTATION_REPLACABLE,
                                           },
                                           )
        position_dependent_keys.add(DICT_POSITION_DEPENDENT)

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
        if old_worker is not None:
            if keep_old_buff:
                # then clear the new buffer and update it with old_worker buffer
                self.zoo[pop_idx].clear_worker_buffer(worker_key=str(local_idx),
                                                      WorkerClass=None,
                                                      )
            self.zoo[pop_idx].update_worker_buffer(local_worker=old_worker,
                                                   worker_key=str(local_idx),
                                                   WorkerClass=None,
                                                   )
        if elo is not None:
            global_idx = self.pop_index_to_index(pop_local_idx=pop_local_idx)
            # replace elo as well
            self.captian_elos[global_idx] = elo

    def save_zoo(self, save_dir):
        """
        saves all zoo cages to specified dir
        Args:
            save_dir: dir to save to
        """
        for zoo_cage in self.zoo:
            zoo_cage.save(save_dir=os.path.join(save_dir, os.path.basename(zoo_cage.zoo_dir)))

    def load_zoo(self, save_dir):
        """
        loads all zoo cages from specified dir
        Args:
            save_dir: dir to save to
        """
        for zoo_cage in self.zoo:
            zoo_cage.load(save_dir=os.path.join(save_dir, os.path.basename(zoo_cage.zoo_dir)))

    def clear_zoo(self):
        for zoo_cage in self.zoo:
            zoo_cage.clear()
        shutil.rmtree(self.zoo_dir)

    def age_up_all_agents(self):
        for pop_idx, popsize in enumerate(self.population_sizes):
            for local_idx in range(popsize):
                self.age_up_agent(pop_idx=pop_idx, local_idx=local_idx)

    def age_up_agent(self, pop_idx, local_idx):
        info = self.get_info(pop_local_idx=(pop_idx, local_idx))
        info[DICT_AGE] = info.get(DICT_AGE, 0) + 1
        self.zoo[pop_idx].overwrite_info(key=str(local_idx), info=info)


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


    def env_constructor():
        return tictactoe_v3.env()


    capzoo = PettingZooCaptianCoevolution(env_constructor=env_constructor,
                                          outcome_fn=MaxOutcome(),
                                          population_sizes=[3, 4, 5],
                                          team_trainer=TeamTrainer(num_agents=3 + 4 + 5),
                                          worker_constructors=[lambda _, env: (WorkerDQN(policy=MlpPolicy,
                                                                                         env=env), {})
                                                               for _ in range(3)],
                                          zoo_dir=os.path.join(DIR, 'data', 'coevolver_zoo_test'),
                                          )
    capzoo.clear_zoo()
