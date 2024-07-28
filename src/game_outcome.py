import torch
from src.zoo_cage import ZooCage
from src.utils.dict_keys import (DICT_IS_WORKER,
                                 DICT_TRAIN,
                                 DICT_COLLECT_ONLY,
                                 DICT_SAVE_BUFFER,
                                 DICT_SAVE_CLASS,
                                 )


class PlayerInfo:
    """
    structure for player observations of a multi-agent game
    """

    def __init__(self, obs_preembed=None, obs_mask=None):
        """
        Args:
            obs_preembed: (S,*) array of a sequence of observations
                None if no observations
            obs_mask: (S,) boolean array of which observations to mask
                None if no mask
        """
        self.obs_preembed = obs_preembed
        self.obs_mask = obs_mask
        self.S = 0 if obs_preembed is None else obs_preembed.shape[0]

    def clone(self):
        return PlayerInfo(obs_preembed=None if self.obs_preembed is None else self.obs_preembed.clone(),
                          obs_mask=None if self.obs_mask is None else self.obs_mask.clone(),
                          )

    def union_obs(self,
                  other_player_info,
                  combine=None,
                  ):
        """
        creates clone, combining observations with other_obs and other_mask

        Args:
            other_obs_preembed: None or  (S',*) sequence of observations to add
                (* must be of same shape as self.obs_preembed's (S,*))
            other_mask: None or (S',) boolean array of whether to mask
            combine: whether to combine or just return updates
                if None, uses chain_observations
        Returns:
            concatenates the two preembeddings and masks
            either outputs an outcome with None, None or
                (S+S', *) concatenated output, (S+S',) boolean mask
                or other_obs_preembed and other_mask (if not chain_observations)
        """
        other_obs_preembed = other_player_info.obs_preembed
        other_mask = other_player_info.obs_mask
        if not combine:
            return PlayerInfo(obs_preembed=other_obs_preembed,
                              obs_mask=other_mask,
                              )
        Sp = 0 if other_obs_preembed is None else other_obs_preembed.shape[0]
        Spp = self.S + Sp

        if Spp == 0:
            # in this case, there are no observations and no masks, so just return empty observation
            return PlayerInfo()

        if (self.obs_mask is None) and (other_mask is None):
            new_mask = None
        else:
            new_mask = torch.zeros(Spp, dtype=torch.bool)
            if self.obs_mask is not None:
                new_mask[:self.S] = self.obs_mask
            if other_mask is not None:
                new_mask[self.S:] = other_mask

        if (self.obs_preembed is not None) and (other_obs_preembed is not None):
            # if both are not None, we append the observations
            # size (S+S', *)
            new_preembed = torch.cat((self.obs_preembed, other_obs_preembed), dim=0)
        else:
            # in this case, one is empty, the other has observations
            # just return the one that is nonempty
            if self.obs_preembed is not None:
                new_preembed = self.obs_preembed.clone()
            else:
                new_preembed = other_obs_preembed
        return PlayerInfo(
            obs_preembed=new_preembed,
            obs_mask=new_mask,
        )

    def __str__(self):
        return ('PlayerInfo(' +
                'obs_preembed:' + str(self.obs_preembed) + '; ' +
                'obs_mask:' + str(self.obs_mask) + '; ' +
                ')')


class OutcomeFn:
    """
    structure for calculating outcomes of a team game
    gives an outcome score for each team, all outcome scores are non-negative and sum to 1
        usually 1 is win, 0.5 is tie (for two team games) and 0 is loss
    """

    def get_outcome(self, team_choices, train_infos=None, env=None):
        """
        Args:
            team_choices: K-tuple of teams, each team is an array of players
            train_infos: either None or array of same shape as team_choices
                each element is dictionary of training settings for corresponding agent
            env: envionrment to use default None

        Returns: list corresponding to teams
            [
                outcome score,
                list corresponding to players of PlayerInfo(
                    obs_preembed=player observation (None or size (S,*) seq of observations);
                    obs_mask=observation mask (None or size (S,) boolean array of which items to mask;
                    )
            ]
        """
        raise NotImplementedError


class PettingZooOutcomeFn(OutcomeFn):
    """
    outcome function which loads rl agents saved as files in specified directory
    """

    def set_zoo(self, zoo: [ZooCage]):
        """
        Args:
            zoo: list of ZooCages
        """
        self.zoo = zoo

    def set_index_conversion(self, index_conversion):
        """
        sets a function converting indices to (zoo index, population index)
        Args:
            index_conversion: (int -> (int, int)); (global index -> (zoo index, population index))
        """
        self.index_conversion = index_conversion

    def index_to_agent(self, idx, training_dict):
        """
        takes index and agent training dict and returns agent
            assumes self.set_zoo and self.set_index_conversion have already been called
        Args:
            idx: global index of agent
            training_dict: training dictionary associated with agent
                includes whether agent is captian, whether there are repeats, etc

        Returns:

        """
        pop_idx, local_idx = self.index_conversion(idx)
        collect_only = training_dict.get(DICT_COLLECT_ONLY, False)
        agent, saved_info = self.zoo[pop_idx].load_animal(key=str(local_idx),
                                                          load_buffer=not collect_only,
                                                          )
        return agent

    def get_outcome(self, team_choices, train_infos=None, env=None):
        """
        Args:
            team_choices: K-tuple of teams, each team is an array of players
                players are indices corresponding
            train_infos: either None or array of same shape as team_choices
                each element is dictionary of training settings for corresponding agent

                relevant keys:
                    DICT_TRAIN: whether the agent should be trained
                    DICT_COLLECT_ONLY: whether the agent should only collect data and update main agent's buffer
                    DICT_SAVE_BUFFER: whether to save buffer of agent (should probably be unspecified or true)
                    DICT_SAVE_CLASS: whether to save class of agent (should probably be unspecified or true)

        Returns: list corresponding to teams
            [
                outcome score,
                list corresponding to players of PlayerInfo(
                    obs_preembed=player observation (None or size (S,*) seq of observations);
                    obs_mask=observation mask (None or size (S,) boolean array of which items to mask;
                    )
            ]
        """
        if train_infos is None:
            train_infos = [[dict() for _ in team] for team in team_choices]

        agent_choices = [
            [self.index_to_agent(member.item(), member_training) for member, member_training in zip(*t)]
            for t in zip(team_choices, train_infos)]
        index_choices = [[member.item() for member in t] for t in team_choices]
        out = self._get_outcome_from_agents(agent_choices=agent_choices,
                                            index_choices=index_choices,
                                            train_infos=train_infos,
                                            env=env,
                                            )
        self._save_agents(agent_choices=agent_choices,
                          index_choices=index_choices,
                          train_infos=train_infos,
                          )
        return out

    def _save_agents(self, agent_choices, index_choices, train_infos):
        for t in zip(agent_choices, index_choices, train_infos):
            for agent, global_idx, train_dict in zip(*t):
                pop_idx, local_idx = self.index_conversion(global_idx)
                cage: ZooCage = self.zoo[pop_idx]
                if train_dict.get(DICT_IS_WORKER, True):
                    if train_dict.get(DICT_COLLECT_ONLY, False):
                        cage.update_worker_buffer(local_worker=agent,
                                                  worker_key=str(local_idx),
                                                  WorkerClass=None,
                                                  )
                    elif train_dict.get(DICT_TRAIN, True):
                        cage.overwrite_worker(worker=agent,
                                              worker_key=str(local_idx),
                                              save_buffer=train_dict.get(DICT_SAVE_BUFFER, True),
                                              save_class=train_dict.get(DICT_SAVE_CLASS, True),
                                              worker_info=train_dict,
                                              )
                cage.overwrite_info(key=str(local_idx), info=train_dict)

    def _get_outcome_from_agents(self, agent_choices, index_choices, train_infos, env):
        """
        from workers, indices, and training info, evaluates the teams in a pettingzoo enviornment and returns
            the output for self.get_outcome
        also should train agents as specified by train_info
        also should save agents as specified by train_info
        Args:
            agent_choices: agents to use for training
            index_choices: global indices of agents
            train_infos: info dicts, same shape as agetns
            env: env to use
        Returns:
            same output as self.get_outcome
            list corresponding to teams
            [
                outcome score,
                list corresponding to players of PlayerInfo(
                    obs_preembed=player observation (None or size (S,*) seq of observations);
                    obs_mask=observation mask (None or size (S,) boolean array of which items to mask;
                    )
            ]
        """
        raise NotImplementedError
