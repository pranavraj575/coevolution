import torch
from src.zoo_cage import ZooCage
from src.utils.dict_keys import (DICT_IS_WORKER,
                                 DICT_TRAIN,
                                 DICT_COLLECT_ONLY,
                                 DICT_SAVE_BUFFER,
                                 DICT_SAVE_CLASS,
                                 )
from collections import defaultdict


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
                  combine=True,
                  ):
        """
        creates clone, combining observations with other_obs and other_mask

        Args:
            other_player_info: other player info with obs and mask either None or shape (S',*) and (S',)
                if self.obs_preembed has shape (S,*), the two * dimensions must be the same
            combine: whether to combine or just return updates
        Returns:
            concatenates the two preembeddings and masks
            either outputs an outcome with None, None or
                (S+1+S', *) concatenated output, (S+1+S',) boolean mask
                 an extra masked element is added in the middle to denote the division
                or other_obs_preembed and other_mask (if not chain_observations)
        """
        other_obs_preembed = other_player_info.obs_preembed
        other_mask = other_player_info.obs_mask
        if not combine:
            return PlayerInfo(obs_preembed=other_obs_preembed,
                              obs_mask=other_mask,
                              )
        Spp = self.S + 1 + other_player_info.S

        if Spp == 1:
            # in this case, there are no observations and no masks, so just return empty observation
            return PlayerInfo()

        if (self.obs_preembed is not None) and (other_obs_preembed is not None):
            # if both are not None, we append the observations,  size (S+1+S', *)

            # divider is shape (1,*)
            divider = torch.zeros_like(self.obs_preembed[:1])
            new_preembed = torch.cat((self.obs_preembed, divider, other_obs_preembed), dim=0)

            # we must also set the mask
            # by default mask nothing except the divider
            new_mask = torch.zeros(Spp, dtype=torch.bool)
            new_mask[self.S] = True  # always mask the division

            if self.obs_mask is not None:
                new_mask[:self.S] = self.obs_mask
            if other_mask is not None:
                new_mask[self.S + 1:] = other_mask
        else:
            # in this case, one is empty, the other has observations
            # just return the one that is nonempty
            if self.obs_preembed is not None:
                new_preembed = self.obs_preembed.clone()
                new_mask = self.obs_mask
            else:
                new_preembed = other_obs_preembed
                new_mask = other_mask
        return PlayerInfo(
            obs_preembed=new_preembed,
            obs_mask=new_mask,
        )

    def __str__(self):
        return ('PlayerInfo(' +
                'obs_preembed:' + str(self.obs_preembed) + '; ' +
                'obs_mask:' + str(self.obs_mask) + '; ' +
                ')')

    def __eq__(self, other):
        if self.obs_preembed is None or other.obs_preembed is None:
            return (self.obs_preembed is None) and (other.obs_preembed is None)

        mask_equality = (
                (self.obs_mask is None and other.obs_mask is None) or
                (self.obs_mask is None and torch.sum(other.obs_mask) == 0) or
                (other.obs_mask is None and torch.sum(self.obs_mask) == 0) or
                torch.equal(self.obs_mask, other.obs_mask)
        )
        return torch.equal(self.obs_preembed, other.obs_preembed) and mask_equality


class OutcomeFn:
    """
    structure for calculating outcomes of a team game
    gives an outcome score for each team, all outcome scores are non-negative and sum to 1
        usually 1 is win, 0.5 is tie (for two team games) and 0 is loss
    """

    def get_outcome(self, team_choices, updated_train_infos=None, env=None):
        """
        Args:
            team_choices: K-tuple of teams, each team is an array of players
            updated_train_infos: either None or array of same shape as team_choices
                each element is dictionary of training settings for corresponding agent
                this is updated with current experiment info (i.e. whether agent is unique, a captian, etc.)
            env: envionrment to use default None

        Returns: list corresponding to teams
            [
                outcome score,
                list corresponding to players of PlayerInfo(
                    obs_preembed=player observation (None or size (S,*) seq of observations);
                    obs_mask=observation mask (None or size (S,) boolean array of which items to mask;
                    )
                list can be empty, this will correspond to an empty observation
            ]
        """
        raise NotImplementedError

    def pop_local_mem(self):
        """
        returns local memory and clears it
        Returns: dict (global idx -> [(trained agent, updated agent info)]
            if the same agent appears multiple times, the respective list will have multiple elements
        """
        return None


class PettingZooOutcomeFn(OutcomeFn):
    """
    outcome function which loads rl agents saved as files in specified directory
    """

    def __init__(self):
        super().__init__()
        self.local_mem = defaultdict(lambda: [])

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

    def get_outcome(self, team_choices, updated_train_infos=None, env=None):
        """
        Args:
            team_choices: K-tuple of teams, each team is an array of players
                players are indices corresponding
            updated_train_infos: either None or array of same shape as team_choices
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
        if updated_train_infos is None:
            updated_train_infos = [[dict() for _ in team] for team in team_choices]

        agent_choices = [
            [self.index_to_agent(member.item(), member_training) for member, member_training in zip(*t)]
            for t in zip(team_choices, updated_train_infos)]
        index_choices = [[member.item() for member in t] for t in team_choices]
        out = self._get_outcome_from_agents(agent_choices=agent_choices,
                                            index_choices=index_choices,
                                            updated_train_infos=updated_train_infos,
                                            env=env,
                                            )
        self._save_agents_to_local_mem(agent_choices=agent_choices,
                                       index_choices=index_choices,
                                       updated_train_infos=updated_train_infos,
                                       )
        return out

    def pop_local_mem(self):
        """
        returns local memory and clears it
        Returns: dict (global idx -> [(trained agent, updated agent info)]
            if the same agent appears multiple times, the respective list will have multiple elements
        """
        local_local_mem = self.local_mem
        self.local_mem = defaultdict(lambda: [])
        return local_local_mem

    def _save_agents_to_local_mem(self, agent_choices, index_choices, updated_train_infos):
        """
        saves trained agents to local memory for another class to pop and save
        Args:
            agent_choices:
            index_choices:
            updated_train_infos:
        Returns:
        """
        for t in zip(agent_choices, index_choices, updated_train_infos):
            for agent, global_idx, updated_train_dict in zip(*t):
                self.local_mem[global_idx].append((agent, updated_train_dict))

    def _get_outcome_from_agents(self, agent_choices, index_choices, updated_train_infos, env):
        """
        from workers, indices, and training info, evaluates the teams in a pettingzoo enviornment and returns
            the output for self.get_outcome
        also should train agents as specified by train_info
        also should save agents as specified by train_info
        Args:
            agent_choices: agents to use for training
            index_choices: global indices of agents
            updated_train_infos: info dicts, same shape as agetns
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


if __name__ == '__main__':
    test0 = PlayerInfo()
    test = PlayerInfo(obs_preembed=torch.rand((1, 2, 3)))
    test2 = PlayerInfo(obs_preembed=torch.rand((2, 2, 3)))
    print(test0)
    print(test)
    print(test0.union_obs(test))
    print(test.union_obs(test))
    print(test.union_obs(test2))
    print(test0 == test)
    print(test0.union_obs(test) == test)
    print(test.union_obs(test0) == test)
