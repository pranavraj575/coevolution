import numpy as np
import torch

from src.coevolver import PettingZooCaptianCoevolution
from BERTeam.trainer import TeamTrainer
from experiments.pyquaticus_utils.agent_aggression import all_agent_aggression, default_potential_opponents

from src.utils.dict_keys import (DICT_AGE,
                                 DICT_TRAIN,
                                 DICT_MUTATION_REPLACABLE,
                                 DICT_IS_WORKER,
                                 DICT_CLONABLE,
                                 DICT_CLONE_REPLACABLE,
                                 DICT_KEEP_OLD_BUFFER,
                                 DICT_UPDATE_WITH_OLD_BUFFER,
                                 TEMP_DICT_CAPTIAN,
                                 DICT_EPOCH_LAST_UPDATED,
                                 )

DICT_EPOCH_LAST_BEHAVIOR_UPDATE = 'epoch_last_behavior_update'
DICT_BEHAVIOR_VECTOR = 'behavior_vector'


class ComparisionExperiment(PettingZooCaptianCoevolution):
    def __init__(self,
                 env_constructor,
                 outcome_fn_gen,
                 population_sizes,
                 storage_dir,
                 worker_constructors,
                 MCAA_mode,
                 games_per_epoch,
                 MCAA_fitness_update=.1,
                 member_to_population=None,
                 team_trainer: TeamTrainer = None,
                 team_sizes=(1, 1),
                 elo_conversion=400/np.log(10),
                 captian_elo_update=32*np.log(10)/400,
                 team_member_elo_update=0*np.log(10)/400,
                 reinit_agents=True,
                 mutation_prob=.001,
                 protect_new=20,
                 protect_elite=1,
                 clone_replacements=None,
                 team_idx_to_agent_id=None,
                 processes=0,
                 temp_zoo_dir=None,
                 max_steps_per_ep=float('inf'),
                 local_collection_mode=True,
                 ):
        super().__init__(
            env_constructor=env_constructor,
            outcome_fn_gen=outcome_fn_gen,
            population_sizes=population_sizes,
            storage_dir=storage_dir,
            worker_constructors=worker_constructors,
            member_to_population=member_to_population,
            team_trainer=team_trainer,
            team_sizes=team_sizes,
            elo_conversion=elo_conversion,
            captian_elo_update=captian_elo_update,
            team_member_elo_update=team_member_elo_update,
            reinit_agents=reinit_agents,
            mutation_prob=mutation_prob,
            protect_new=protect_new,
            protect_elite=protect_elite,
            clone_replacements=clone_replacements,
            team_idx_to_agent_id=team_idx_to_agent_id,
            processes=processes,
            temp_zoo_dir=temp_zoo_dir,
            max_steps_per_ep=max_steps_per_ep,
            depth_to_retry_result=None,
            local_collection_mode=local_collection_mode,
        )
        self.MCAA = MCAA_mode
        self.MCAA_fitness_update = MCAA_fitness_update
        self.games_per_epoch = games_per_epoch
        if self.MCAA:
            # we will use elos as the fitness, representing win probability against random team
            # we will start them off as all .5
            self.set_elos(.5*torch.ones_like(self.elos))

    # TODO: make a mode that does not use captians to generate teams
    #  'team captian off' setting
    #  use this for MCAA
    #  this will be used in preepisode dict generation
    def complete_epoch_and_extra_training(self, all_items_to_save, epoch_info, **kwargs):
        # TODO: keep track of individual fitness, either through win rate or normal captian elo

        if not self.MCAA:
            return super().complete_epoch_and_extra_training(all_items_to_save=all_items_to_save,
                                                             epoch_info=epoch_info,
                                                             **kwargs,
                                                             )
        else:
            # TODO: IF DOING MCAA MAINLAND
            #  view the winners of first round, remake a parining with all winning teams
            #  dont do this if there are less than k teams left
            #  also maybe make new teams by switching out some stuff
            #  also maybe just do this anyway for consistency
            return super().complete_epoch_and_extra_training(all_items_to_save=all_items_to_save,
                                                             epoch_info=epoch_info,
                                                             **kwargs,
                                                             )

    def update_results(self, items_to_save):
        # TODO: update individual elos or fitnesses here
        #  i think it makes sense to use captian elos when using BERTeam, and doing win rate stuff otherwise
        if not self.MCAA:
            return super().update_results(items_to_save=items_to_save)
        else:
            # save trained agents, if applicable
            self.save_trained_agents(items_to_save['outcome_local_mem'])
            for (captian,
                 team,
                 (team_outcome, _),
                 ) in zip(items_to_save['captian_choices'],
                          items_to_save['teams'],
                          items_to_save['team_outcomes']
                          ):
                for member in team:
                    self.elos[member] = (self.MCAA_fitness_update*team_outcome +
                                         (1 - self.MCAA_fitness_update)*self.elos[member])

    def epoch(self,
              rechoose=True,
              save_epoch_info=True,
              pre_ep_dicts=None,
              update_epoch_infos=True,
              noise_model=None,
              depth=0,
              known_obs=(None, None),
              save_trained_agents=True,
              save_into_team_buffer=True,
              **kwargs,
              ):
        if not self.MCAA:
            return super().epoch(
                rechoose=rechoose,
                save_epoch_info=save_epoch_info,
                pre_ep_dicts=pre_ep_dicts,
                update_epoch_infos=update_epoch_infos,
                noise_model=noise_model,
                depth=depth,
                known_obs=known_obs,
                save_trained_agents=save_trained_agents,
                save_into_team_buffer=save_into_team_buffer,
                **kwargs,
            )

        # TODO: pass through preepisode dicts using MCAA to generate teams
        #  also maybe do the pre episode generation to make tournament form
        pre_ep_dicts = [self.captianless_pre_episode() for _ in range(self.games_per_epoch)]

        return super().epoch(
            rechoose=rechoose,
            save_epoch_info=save_epoch_info,
            pre_ep_dicts=pre_ep_dicts,
            update_epoch_infos=update_epoch_infos,
            noise_model=noise_model,
            depth=depth,
            known_obs=known_obs,
            save_trained_agents=save_trained_agents,
            save_into_team_buffer=save_into_team_buffer,
            **kwargs,
        )

    def captianless_pre_episode(self):
        """
        for use when not using BERTeam
        generates random teams to play against each other
        Returns:
        """
        print('generating preep')
        episode_info = dict()

        teams = []
        for team_idx, team_size in enumerate(self.team_sizes):
            team = self.team_trainer.create_teams(T=team_size, N=1).detach().flatten()
            teams.append(team)

        episode_info['teams'] = tuple(team.numpy() for team in teams)
        outcome_fn = self.create_outcome_fn()
        train_infos = []
        for team_idx, team in enumerate(teams):
            tinfo = []
            for i, member in enumerate(team):
                global_idx = member.item()
                captian_pop_idx, local_idx = self.index_to_pop_index(global_idx=global_idx)

                info = self.get_info(pop_local_idx=(captian_pop_idx, local_idx))
                info[TEMP_DICT_CAPTIAN] = False
                tinfo.append(info)
            train_infos.append(tinfo)

        env = self.env_constructor(train_infos)

        return {'ident': 0,
                'teams': teams,
                'agents': None,
                'train_infos': train_infos,
                'env': env,
                'outcome_fn': outcome_fn,
                'episode_info': episode_info,
                'captian_choices': [None for _ in range(len(teams))],
                }

    def create_team(self,
                    team_idx,
                    captian=None,
                    obs_preembed=None,
                    obs_mask=None,
                    noise_model=None,
                    team_noise_model=None,
                    ):
        if self.MCAA:
            team_size = self.team_sizes[team_idx]
            team = self.team_trainer.create_teams(T=team_size,
                                                  N=1,
                                                  noise_model=noise_model,
                                                  )
            team = team.detach().flatten()
            return team, None
        else:
            return super().create_team(team_idx=team_idx,
                                       captian=captian,
                                       obs_preembed=obs_preembed,
                                       obs_mask=obs_mask,
                                       noise_model=noise_model,
                                       team_noise_model=team_noise_model,
                                       )


class PZCC_MAPElites(ComparisionExperiment):
    def __init__(self,
                 env_constructor,
                 outcome_fn_gen,
                 population_sizes,
                 storage_dir,
                 worker_constructors,
                 MCAA_mode,
                 games_per_epoch,
                 MCAA_fitness_update=.1,
                 default_behavior_radius=1.,
                 member_to_population=None,
                 team_trainer: TeamTrainer = None,
                 team_sizes=(1, 1),
                 elo_conversion=400/np.log(10),
                 captian_elo_update=32*np.log(10)/400,
                 team_member_elo_update=0*np.log(10)/400,
                 reinit_agents=True,
                 mutation_prob=.001,
                 protect_new=20,
                 protect_elite=1,
                 clone_replacements=None,
                 team_idx_to_agent_id=None,
                 processes=0,
                 temp_zoo_dir=None,
                 max_steps_per_ep=float('inf'),
                 local_collection_mode=True,
                 ):
        super().__init__(
            env_constructor=env_constructor,
            outcome_fn_gen=outcome_fn_gen,
            population_sizes=population_sizes,
            storage_dir=storage_dir,
            MCAA_mode=MCAA_mode,
            MCAA_fitness_update=MCAA_fitness_update,
            games_per_epoch=games_per_epoch,
            worker_constructors=worker_constructors,
            member_to_population=member_to_population,
            team_trainer=team_trainer,
            team_sizes=team_sizes,
            elo_conversion=elo_conversion,
            captian_elo_update=captian_elo_update,
            team_member_elo_update=team_member_elo_update,
            reinit_agents=reinit_agents,
            mutation_prob=mutation_prob,
            protect_new=protect_new,
            protect_elite=protect_elite,
            clone_replacements=clone_replacements,
            team_idx_to_agent_id=team_idx_to_agent_id,
            processes=processes,
            temp_zoo_dir=temp_zoo_dir,
            max_steps_per_ep=max_steps_per_ep,
            local_collection_mode=local_collection_mode,
        )
        self.default_behavior_radius = default_behavior_radius

    def behavior_projection(self):
        """
        projects all agents into behavior space
        Returns: (popsize, k), behavior vector for each member
        """
        print('obtaining behavior')
        # TODO: send in aggressions or some other metric
        agent_indices = []
        aggressions = [0 for _ in range(self.N)]
        for i in range(self.N):
            info = self.load_animal(self.index_to_pop_index(i), load_buffer=False)[1]
            last_update = info.get(DICT_EPOCH_LAST_UPDATED, 0)
            last_behavior_update = info.get(DICT_EPOCH_LAST_BEHAVIOR_UPDATE, -1)
            if last_behavior_update < last_update:
                agent_indices.append(i)
            else:
                aggressions[i] = info[DICT_BEHAVIOR_VECTOR]

        agents = (self.load_animal(self.index_to_pop_index(i))[0] for i in agent_indices)

        (att_easy, att_mid, att_hard,
         def_easy, def_mid, def_hard,
         rand) = default_potential_opponents(env_constructor=self.env_constructor)
        aggression_updates = all_agent_aggression(agents,
                                                  env_constructor=self.env_constructor,
                                                  team_size=self.team_sizes[0],
                                                  processes=self.processes,
                                                  potential_opponents=[att_easy, att_hard, def_easy, def_hard],
                                                  )
        for i, agg in zip(agent_indices, aggression_updates):
            aggressions[i] = agg

            info = self.load_animal(self.index_to_pop_index(i), load_buffer=False)[1]
            info[DICT_EPOCH_LAST_BEHAVIOR_UPDATE] = self.epochs
            info[DICT_BEHAVIOR_VECTOR] = agg
            pop_idx, local_idx = self.index_to_pop_index(i)
            cage = self.zoo[pop_idx]
            cage.overwrite_info(key=str(local_idx), info=info)
        return torch.tensor(aggressions).reshape(self.N, 1)

    def get_fitness_estimate(self, elos, locations, behavior_radius=None):
        if behavior_radius is None:
            behavior_radius = self.default_behavior_radius
        loc_dim = locations.shape[1]
        # take the average over nearby values, not including self
        loc_matrix = torch.linalg.norm(locations.view(-1, 1, loc_dim) -
                                       locations.view(1, -1, loc_dim),
                                       dim=-1,
                                       ) + torch.diag(torch.inf*torch.ones(locations.shape[0]))
        valid_locs = loc_matrix <= behavior_radius
        valid_locs = valid_locs/torch.sum(valid_locs, dim=0, keepdim=True)
        estimates = torch.matmul(elos.view(1, -1), valid_locs)
        return estimates.flatten()

    def conservative_breed(self,
                           number_to_replace: [int],
                           base_elo=0.,
                           force_replacements=False,
                           ):
        # TODO: do the MAP-Elites thing and replace bad ones (comparative) with clones of good ones (comparative)
        #  may also add unique to the clonables

        all_behavior_vectors = self.behavior_projection()
        behavior_dim = all_behavior_vectors.shape[1]

        breed_dic = {'number_replaced': [0 for _ in self.population_sizes],
                     'target_agents': [],
                     'target_elos': [],
                     'cloned_agents': [],
                     'cloned_elos': [],
                     }
        for pop_idx in torch.randperm(len(self.population_sizes)):
            popsize = self.population_sizes[pop_idx]
            cum_popsize = self.cumsums[pop_idx]
            if type(number_to_replace) == int:
                if number_to_replace <= 0:
                    continue
            elif number_to_replace[pop_idx] <= 0:
                continue

            behavior_vectors = all_behavior_vectors[cum_popsize: cum_popsize + popsize]
            behavior_matrix = torch.linalg.norm(behavior_vectors.view(-1, 1, behavior_dim) -
                                                behavior_vectors.view(1, -1, behavior_dim),
                                                dim=-1,
                                                ) + torch.diag(torch.inf*torch.ones(popsize))
            behavior_radius = self.default_behavior_radius
            # ignore unique values
            non_unique_indices = [cum_popsize + i for i in range(popsize)
                                  if torch.min(behavior_matrix[i]) < behavior_radius]

            # pick the agents to potentially clone
            candidate_clone_idxs = list(self._get_valid_idxs(validity_fn=
                                                             lambda info:
                                                             info.get(DICT_CLONABLE, True),
                                                             indices=range(cum_popsize, cum_popsize + popsize),
                                                             )
                                        )
            if not candidate_clone_idxs:
                # no clonable agents
                continue
            # pick the agents to potentially replace with a clone
            if force_replacements:
                candidate_target_idxs = list(
                    self._get_valid_idxs(validity_fn=lambda info: info.get(DICT_CLONE_REPLACABLE, True) and
                                                                  (info.get(DICT_AGE, 0) > self.protect_new),
                                         indices=non_unique_indices,
                                         ))
            else:
                candidate_target_idxs = non_unique_indices

            # remove the elite from potential targets
            elite = cum_popsize + torch.topk(self.elos[cum_popsize: cum_popsize + popsize],
                                             k=self.protect_elite
                                             ).indices
            candidate_target_idxs = [idx for idx in candidate_target_idxs if idx not in elite]
            # can replace at most this number
            number_to_replace[pop_idx] = min(len(candidate_target_idxs), number_to_replace[pop_idx])
            if not candidate_target_idxs:
                continue

            original_target_elos = self.elos[candidate_target_idxs]
            target_behaviors = all_behavior_vectors[candidate_target_idxs]
            elo_estimate = self.get_fitness_estimate(elos=original_target_elos,
                                                     locations=target_behaviors,
                                                     behavior_radius=behavior_radius,
                                                     )
            comp_target_elos = original_target_elos - elo_estimate

            # distribution of agents based on how bad they are COMPARATIVELY
            candidate_target_dist = self.get_inverted_distribution(elos=comp_target_elos)
            # pick a random subset of target agents to replace based on this distribution
            target_idx_idxs = torch.multinomial(candidate_target_dist, number_to_replace[pop_idx], replacement=False)
            # these are the global indexes of the targets
            target_global_idxs = [candidate_target_idxs[target_idx_idx] for target_idx_idx in target_idx_idxs]
            target_elos = [self.elos[target_global_idx].item() for target_global_idx in target_global_idxs]
            target_comparative_elos = [comp_target_elos[target_idx_idx].item() for target_idx_idx in target_idx_idxs]

            # now pick which agents to clone based on elo (NON COMPARATIVE)
            candidate_clone_elos = self.elos[candidate_clone_idxs]
            clone_dist = torch.softmax(candidate_clone_elos, dim=-1)
            # sample from this distribution with replacement
            clone_idx_idxs = list(torch.multinomial(clone_dist, len(target_global_idxs), replacement=True))
            # element clone_idx_idx in clone_idx_idxs denotes that candidate_clone_idxs[clone_idx_idx] should be cloned
            # also candidate_clone_idxs[clone_idx_idx] has elo clone_elos[clone_idx_idx]

            # global indexes of clones, as well as elos of the clones
            clone_global_idxs = [candidate_clone_idxs[clone_idx_idx] for clone_idx_idx in clone_idx_idxs]
            clone_elos = [candidate_clone_elos[clone_idx_idx].item() for clone_idx_idx in clone_idx_idxs]

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
                    breed_dic['number_replaced'][pop_idx] += 1
                    breed_dic['target_agents'].append(target_global_idx)
                    breed_dic['cloned_agents'].append(clone_global_idx)
                    breed_dic['cloned_elos'].append(clone_elo)
                    breed_dic['target_elos'].append(target_elo)
                    # note: it is probably necessary to save clone_elo and target_elo lists beforehand as self.captain_elos
                    # are being reassigned with self.replace_agent

            if type(number_to_replace) == int:
                number_to_replace -= breed_dic['number_replaced'][pop_idx]
        breed_dic['based_elos'] = base_elo
        if base_elo is not None:
            self.rebase_elos(base_elo=base_elo)
        self.age_up_all_agents()
        return breed_dic


if __name__ == '__main__':
    import os

    from experiments.pyquaticus_utils.wrappers import MyQuaticusEnv
    from experiments.pyquaticus_utils.reward_fns import custom_rew2
    from repos.pyquaticus.pyquaticus.config import config_dict_std
    from experiments.pyquaticus_utils.outcomes import CTFOutcome

    from unstable_baselines3 import WorkerPPO
    from unstable_baselines3.ppo import MlpPolicy as PPOMlp
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    team_size = 2

    reward_config = {i: custom_rew2 for i in range(2*team_size)}  # Example Reward Config
    config_dict = config_dict_std
    config_dict["max_screen_size"] = (float('inf'), float('inf'))


    def env_constructor(train_infos):
        return MyQuaticusEnv(save_video=False,
                             render_mode=None,
                             reward_config=reward_config,
                             team_size=team_size,
                             config_dict=config_dict,
                             )


    net_arch = [64, 64]
    policy_kwargs = {
        'net_arch': dict(pi=net_arch,
                         vf=net_arch),
    }
    train_info_dict = {DICT_TRAIN: True,
                       DICT_CLONABLE: True,
                       DICT_CLONE_REPLACABLE: True,
                       DICT_MUTATION_REPLACABLE: True,
                       DICT_IS_WORKER: True,
                       }
    create_ppo = lambda i, env: (WorkerPPO(policy=PPOMlp,
                                           env=env,
                                           policy_kwargs=policy_kwargs,
                                           ), train_info_dict.copy()
                                 )
    population_sizes = (2, 2, 2, 2, 2)
    thingy = PZCC_MAPElites(env_constructor=env_constructor,
                            outcome_fn_gen=CTFOutcome,
                            population_sizes=population_sizes,
                            storage_dir=os.path.join('temp', 'subclass_test'),
                            worker_constructors=[create_ppo for _ in population_sizes],
                            MCAA_mode=True,
                            team_sizes=(2, 2),
                            processes=6,
                            games_per_epoch=2,
                            )
    # thingy.conservative_breed(number_to_replace=2)
    start = time.time()
    thingy.epoch(rechoose=False)
    print(time.time() - start)
    thingy.clear()
