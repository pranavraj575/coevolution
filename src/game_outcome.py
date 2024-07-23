import torch


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

    def get_outcome(self, team_choices, train=None):
        """
        Args:
            team_choices: K-tuple of teams, each team is an array of players
            train: either None or boolean array of same shape as team_choices
                whether to train each agent
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
