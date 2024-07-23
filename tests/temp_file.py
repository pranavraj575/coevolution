import torch
def collect_training_data(trainer,
                          outcome,
                          num_members,
                          N,
                          init_teams=None,
                          initial_obs_preembeds=None,
                          initial_obs_masks=None,
                          number_of_loss_rematches=0,
                          number_of_tie_matches=0,
                          chain_observations=True,
                          noise_model=None,
                          obtain_negatives=False,
                          ):
    """
    Args:
        num_members: K-tuple of members for each team
            ie (2,2) is a 2v2 game
        N: number of games to collect
        init_teams: teams to start playing with (if None, chooses N sets of K random teams)
        initial_obs_preembeds: K-tuple of preembddings to use for each team
            either None or list of ((N,S,*) or None)
        initial_obs_masks: K-tuple of masks to use for each team input
            either None or list of (boolean array (N,S) or None)
        number_of_loss_rematches: number of times to build teams to replace losing teams (conditioned on obs)
        number_of_tie_matches: number of times to build teams to replace tied teams (conditioned on obs)
        chain_observations: whether to chain the observations across different games
        noise_model: noise model to select teams with
        obtain_negatives: whether to grab 'negative' examples (losing teams with negative scalar returns)
            this will always be false on rematches
        outcome:
            Returns the winning agents, as well as the contexts
            Args:
                teams: K-tuple of (N, T_i) arrays of players, with K being the number of teams (T_1, ..., T_k)
            Returns:
                (
                    win_indices: which games had winners (game number list, indices of winning team), both size (N')
                    pre_embedded_observations: None or (N', S', *) array observed by winning team
                    embedding_mask: None or (N', S') boolean array of which items to mask
                ),
                (
                    loss_indices: which games had losers (game number list, indices of winning team), both size (N'')
                    pre_embedded_observations: None or (N'', S'', *) array observed by losing team
                    embedding_mask: None or (N'', S'') boolean array of which items to mask
                ),
                (
                    tied_indices: indices that tied, (game number list, indices of tied team), both size (N''')
                    pre_embedded_observations: None or (N''', S''', *) array observed by tied team
                    embedding_mask: None or (N''', S''') boolean array of which items to mask
                )
    Returns:
        iterable of (scalar, observation, winning team, observation_mask) to push into replay buffer
            scalar is 1 for 'good' teams and -1 for 'bad' teams
    """

    def append_obs(init_obs_preembed,
                   update_obs_preembed,
                   init_mask=None,
                   update_mask=None,
                   combine=None):
        """

        Args:
            init_obs_preembed: (N, S, *) preembedding or None
            update_obs_preembed: (N, S', *) preembedding or None
            init_mask: (N, S) boolean mask array or None
            update_mask: (N, S') boolean mask array or None
            combine: whether to combine or just return updates
                if None, uses chain_observations
        Returns:
            concatenates the two preembeddings and masks
            either outputs None, None or
                (N, S+S', *) concatenated output, (N, S+S') boolean mask
                or update obs and update mask (if not chain_observations)
        """
        if combine is None:
            combine = chain_observations

        if not combine:
            return update_obs_preembed, update_mask

        if init_obs_preembed is None and update_obs_preembed is None:
            return (None, None)
        elif (init_obs_preembed is not None) and (update_obs_preembed is not None):
            # if both are not None, we append the observations
            obs_preembed = torch.cat((init_obs_preembed, update_obs_preembed), dim=1)
            if init_mask is None and update_mask is None:
                return obs_preembed, None
            elif (init_mask is not None) and (update_mask is not None):
                return (obs_preembed, torch.cat((init_mask, update_mask), dim=1))
            else:
                N, S = init_obs_preembed.shape
                N, Sp = update_obs_preembed.shape
                mask = torch.zeros((N, S + Sp), dtype=torch.bool)
                if init_mask is not None:
                    mask[:, :S] = init_mask
                else:
                    mask[:, S:] = update_mask
                return (obs_preembed, mask)
        else:
            # in this case, one is empty, the other has observations
            # just return the one that is nonempty
            if init_obs_preembed is not None:
                return init_obs_preembed, init_mask
            else:
                return update_obs_preembed, update_mask

    K = len(num_members)
    if initial_obs_preembeds is None:
        initial_obs_preembeds = [None for _ in range(K)]
    if initial_obs_masks is None:
        initial_obs_masks = [None for _ in range(K)]
    if init_teams is None:
        teams = [trainer.create_masked_teams(T=T_k, N=N) for T_k in num_members]
    else:
        teams = [trainer.create_masked_teams(T=T_k, N=N) if team is None else team
                 for T_k, team in zip(num_members, init_teams)]
    for i in range(K):
        teams[i] = trainer.fill_in_teams(initial_teams=teams[i],
                                         obs_preembed=initial_obs_preembeds[i],
                                         obs_mask=initial_obs_masks[i],
                                         noise_model=noise_model,
                                         )
    win_stuff, lose_stuff, tie_stuff = outcome(teams)
    for k in range(len(num_members)):
        team_k = teams[k]
        init_obs_preembeds_k = initial_obs_preembeds[k]
        init_obs_masks_k = initial_obs_masks[k]

        for (((trials, team_nos), temp_obs, temp_mask),
             (scalar, do_rematch, temp_loss_rematch, temp_tie_rematch)) in [
            (win_stuff, (1., False, number_of_loss_rematches, number_of_tie_matches)),
            (lose_stuff, (-1., True, number_of_loss_rematches - 1, number_of_tie_matches)),
            (tie_stuff, (0., True, number_of_loss_rematches, number_of_tie_matches - 1)),
        ]:
            if do_rematch and (temp_loss_rematch < 0 or temp_tie_rematch < 0):
                # dont do rematches in this case
                continue

            rematches = []
            rematch_obs_dim = 0
            rematch_obs_shape = None
            rematch_obs_dtype = None

            # trial idx is the index wrt number of winning teams (i.e. first winning team is 0)
            for trial_idx in torch.where(team_nos == k)[0]:
                # 'global' index, index in original enumeration
                global_idx = trials[trial_idx]
                temp_init_obs_preembed = (None if init_obs_preembeds_k is None
                                          else init_obs_preembeds_k[(global_idx,)].unsqueeze(0))

                temp_init_obs_mask = (None if init_obs_masks_k is None
                                      else init_obs_masks_k[(global_idx,)].unsqueeze(0))

                temp_new_obs_preembed = (None if temp_obs is None
                                         else temp_obs[(trial_idx,)].unsqueeze(0))
                temp_new_obs_mask = (None if temp_mask is None
                                     else temp_mask[(trial_idx,)].unsqueeze(0))
                preembed, mask = append_obs(init_obs_preembed=temp_init_obs_preembed,
                                            update_obs_preembed=temp_new_obs_preembed,
                                            init_mask=temp_init_obs_mask,
                                            update_mask=temp_new_obs_mask,
                                            )
                if do_rematch:
                    rematches.append((global_idx, preembed, mask))
                    if preembed is not None:
                        rematch_obs_dim = max(rematch_obs_dim, preembed.shape[1])
                        rematch_obs_shape = preembed.shape[2:]
                        rematch_obs_dtype = preembed.dtype
                if scalar > 0:
                    yield (scalar, preembed, team_k[(global_idx,),], mask)
                if scalar < 0 and obtain_negatives:
                    yield (scalar, preembed, team_k[(global_idx,),], mask)

            if do_rematch and rematches:
                # keep all matchups the same except the index k
                new_init_teams = [team[[global_idx for global_idx, _, _ in rematches],] if i != k else None
                                  for i, team in enumerate(teams)]
                N_p = len(rematches)
                if rematch_obs_dim == 0:
                    relevant_obs_preembed = None
                    relevant_obs_mask = None
                else:
                    temp_shape = [N_p, rematch_obs_dim]
                    for size in rematch_obs_shape:
                        temp_shape.append(size)
                    relevant_obs_preembed = torch.zeros(temp_shape,
                                                        dtype=rematch_obs_dtype)
                    relevant_obs_mask = torch.ones((N_p, rematch_obs_dim), dtype=torch.bool)
                    for i, (_, preembed, mask) in enumerate(rematches):
                        S = preembed.shape[1]
                        relevant_obs_preembed[(i,), :S] = preembed
                        if mask is not None:
                            # in this case, copy mask over
                            relevant_obs_mask[(i,), :S] = mask
                        else:
                            # otherwise, set mask to false here
                            relevant_obs_mask[(i,), :S] = 0
                for item in trainer.collect_training_data(
                        outcome=outcome,
                        num_members=num_members,
                        N=len(rematches),
                        init_teams=new_init_teams,
                        initial_obs_preembeds=[None if i != k else relevant_obs_preembed for i in range(K)],
                        initial_obs_masks=[None if i != k else relevant_obs_mask for i in range(K)],
                        number_of_loss_rematches=temp_loss_rematch,
                        number_of_tie_matches=temp_tie_rematch,
                        chain_observations=chain_observations,
                        noise_model=noise_model,
                        obtain_negatives=False,
                ):
                    yield item