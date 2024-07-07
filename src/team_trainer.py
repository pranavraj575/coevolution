import torch
from torch import nn
from torch.utils.data import DataLoader

from networks.team_builder import TeamBuilder, BERTeam
from networks.input_embedding import DiscreteInputEmbedder, DiscreteInputPosEmbedder, DiscreteInputPosAppender


class TeamTrainer:
    def __init__(self,
                 team_builder: TeamBuilder,
                 ):
        """
        Args:
        """
        self.team_builder = team_builder
        self.MASK = team_builder.berteam.MASK
        self.num_agents = team_builder.berteam.num_agents

        self.optim = torch.optim.Adam(team_builder.parameters())

    def collect_training_data(self,
                              outcome,
                              num_members,
                              N,
                              init_teams=None,
                              initial_obs_preembeds=None,
                              initial_obs_masks=None,
                              number_of_loss_rematches=0,
                              number_of_tie_matches=0,
                              chain_observations=True,
                              noise_model=None
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
            outcome:
                Returns the winning agents, as well as the contexts
                Args:
                    players: K-tuple of (N, T_i) arrays of players, with K being the number of teams (T_1, ..., T_k)
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
            iterable of (winning team, observation, observation_mask) to push into replay buffer
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
            if combine:
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
                    return update_obs_preembed, update_mask
                else:
                    return init_obs_preembed, init_mask

        K = len(num_members)
        if initial_obs_preembeds is None:
            initial_obs_preembeds = [None for _ in range(K)]
        if initial_obs_masks is None:
            initial_obs_masks = [None for _ in range(K)]
        if init_teams is None:
            teams = [self.create_masked_teams(T=T_k, N=N) for T_k in num_members]
        else:
            teams = [self.create_masked_teams(T=T_k, N=N) if team is None else team
                     for T_k, team in zip(num_members, init_teams)]
        for i in range(K):
            teams[i] = self.fill_in_teams(initial_teams=teams[i],
                                          obs_preembed=initial_obs_preembeds[i],
                                          obs_mask=initial_obs_masks[i],
                                          noise_model=noise_model,
                                          )
        print('teams')
        print(teams[0])
        print(teams[1])
        print()
        win_stuff, lose_stuff, tie_stuff = outcome(teams)
        for k in range(len(num_members)):
            team_k = teams[k]
            init_obs_preembeds_k = initial_obs_preembeds[k]
            init_obs_masks_k = initial_obs_masks[k]
            for ((trials, team_nos), temp_obs, temp_mask), (do_rematch, temp_loss_rematch, temp_tie_rematch) in [
                (win_stuff, (False, number_of_loss_rematches, number_of_tie_matches)),
                (lose_stuff, (True, number_of_loss_rematches - 1, number_of_tie_matches)),
                (tie_stuff, (True, number_of_loss_rematches, number_of_tie_matches - 1)),
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
                    else:
                        yield (team_k[global_idx], preembed, mask)

                if do_rematch and rematches:
                    # keep all matchups the same except the index k
                    new_init_teams = [team[[global_idx for global_idx, _, _ in rematches],] if i != k else None
                                      for i, team in enumerate(teams)]
                    N_p = len(rematches)
                    if rematch_obs_dim == 0:
                        relevant_obs_preembed = None
                        relevant_obs_mask = None
                    else:
                        temp_shape=[N_p,rematch_obs_dim]
                        for size in rematch_obs_shape:
                            temp_shape.append(size)
                        relevant_obs_preembed = torch.zeros(temp_shape,
                                                            dtype=rematch_obs_dtype)
                        relevant_obs_mask = torch.ones((N_p, rematch_obs_dim), dtype=torch.bool)
                        for i, (_, preembed, mask) in enumerate(rematches):
                            relevant_obs_preembed[(i,), :preembed.shape[1]] = preembed
                            if mask is not None:
                                relevant_obs_mask[(i,), :mask.shape[1]] = mask
                    for item in self.collect_training_data(
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
                    ):
                        yield item

    def training_step(self,
                      loader: DataLoader,
                      mask_probs=None,
                      replacement_probs=(.8, .1, .1),
                      minibatch=True,
                      ):
        """
        Args:
            loader: contains data of type (obs_preembed, teams, obs_mask)
                obs_preembed and obs_mask can both be torch.nan, they are ignored if this is true
            minibatch: whether to take a step after each batch
        Returns:

        """
        if not minibatch:
            self.optim.zero_grad()
        all_losses = []
        for obs_preembed, teams, obs_mask in loader:
            if torch.is_tensor(obs_preembed) and torch.all(torch.isnan(obs_preembed)):
                obs_preembed = None
            if torch.is_tensor(obs_mask) and torch.all(torch.isnan(obs_mask)):
                obs_mask = None
            if minibatch:
                self.optim.zero_grad()

            losses = self.mask_and_learn(obs_preembed=obs_preembed,
                                         teams=teams,
                                         obs_mask=obs_mask,
                                         mask_probs=mask_probs,
                                         replacement_probs=replacement_probs,
                                         )
            if minibatch:
                self.optim.step()
            all_losses.append(losses)
        if not minibatch:
            self.optim.step()
        return torch.mean(torch.tensor(all_losses))

    def mask_and_learn(self, obs_preembed, teams, obs_mask, mask_probs=None, replacement_probs=(.8, .1, .1)):
        """
        runs learn step on a lot of mask_probabilities
            need to test with a wide range of mask probabilites because for generation, we may start with a lot of masks
        Args:
            obs_preembed: tensor (N, S, *) of input preembeddings, or None if no preembeddings
            teams: tensor (N, T) of teams
            obs_mask: boolean tensor (N, S) of whether to pad each input
            mask_probs: list of proabilities of masking to use
                if None, uses (1/team_size, 2/team_size, ..., 1)
            replacement_probs: proportion of ([MASK], random element, same element) to replace masked elements with
                default (.8, .1, .1) because BERT
        Returns:
            list of crossentropy loss
        """
        if mask_probs is None:
            N, T = teams.shape
            mask_probs = torch.arange(0, T + 1)/T
        losses = []
        for mask_prob in mask_probs:
            loss = self.get_losses(obs_preembed=obs_preembed,
                                   teams=teams,
                                   obs_mask=obs_mask,
                                   mask_prob=mask_prob,
                                   replacement_probs=replacement_probs,
                                   )
            losses.append(loss)
        return torch.mean(torch.tensor(losses))

    def get_losses(self, obs_preembed, teams, obs_mask, mask_prob=.5, replacement_probs=(.8, .1, .1)):
        """
        randomly masks winning team members, runs the transformer token prediction model, and gets crossentropy loss
            of predicting the masked tokens
        Args:
            obs_preembed: tensor (N, S, *) of input preembeddings, or None if no preembeddings
            teams: tensor (N, T) of teams
            obs_mask: boolean tensor (N, S) of whether to pad each input
            mask_prob: proportion of elements to mask (note that we will always mask at least one per batch
            replacement_probs: proportion of ([MASK], random element, same element) to replace masked elements with
                default (.8, .1, .1) because BERT
        Returns:
            crossentropy loss of prediction
        """

        in_teams, mask, mask_indices = self._randomly_mask_teams(teams=teams,
                                                                 mask_prob=mask_prob,
                                                                 replacement_props=replacement_probs,
                                                                 )
        logits = self.team_builder.forward(obs_preembed=obs_preembed,
                                           target_team=in_teams,
                                           obs_mask=obs_mask,
                                           output_probs=True,
                                           pre_softmax=True,
                                           )
        criterion = nn.CrossEntropyLoss()

        loss = criterion(logits[mask_indices], teams[mask_indices])
        loss.backward()
        return loss

    def _randomly_mask_teams(self, teams, mask_prob, replacement_props):
        """
        Args:
            teams: size (N,T) tensor to randomly mask
            mask_prob: proportion of elements to mask (note that we will always mask at least one per batch)
            replacement_props: proportion of ([MASK], random element, same element) to replace masked elements with
                tuple of three floats
        """
        masked_teams = teams.clone()
        N, T = teams.shape

        which_to_mask = torch.bernoulli(torch.ones_like(teams)*mask_prob)
        forced_mask = torch.randint(0, T, (N,))
        which_to_mask[torch.arange(N), forced_mask] = 1

        mask_indices = torch.where(which_to_mask == 1)
        num_masks = len(mask_indices[0])
        what_to_replace_with = torch.multinomial(torch.tensor([replacement_props for _ in range(num_masks)]),
                                                 1).flatten()
        # this will be a tensor of 0s, 1s, and 2s, indicating whether to replace each mask with
        #   [MASK], a random element, or not to replace it

        # map this to -1,-2,-3 respectively
        what_to_replace_with = -1 - what_to_replace_with
        # now map to the actual values to replace with in the array
        what_to_replace_with = torch.masked_fill(what_to_replace_with,
                                                 mask=(what_to_replace_with == -1),
                                                 value=self.MASK,
                                                 )
        rand_rep = torch.where(what_to_replace_with == -2)
        what_to_replace_with[rand_rep] = self.uniform_random_team((len(rand_rep[0]),))

        unchange = torch.where(what_to_replace_with == -3)
        # grab the correct team members from the original array
        what_to_replace_with[unchange] = masked_teams[[dim_idx[unchange] for dim_idx in mask_indices]]

        masked_teams[mask_indices] = what_to_replace_with
        return masked_teams, which_to_mask, mask_indices

    def uniform_random_team(self, shape):
        return torch.randint(0, self.num_agents, shape)

    def create_masked_teams(self, T, N=1):
        """
        Args:
            T: team size
            N: batch size
        Returns:
            an (N,T) tensor of all masked members
        """
        return torch.fill(torch.zeros((N, T), dtype=torch.long), self.MASK)

    def create_nose_model_towards_uniform(self, t):
        """
        creates a noise model that takes in a distribution and does a weighted average with a uniform dist
        Args:
            t: weight to give uniform dist (1.0 replaces the distribution, 0.0 leaves dist unchanged)
        Returns:
            distribution -> distribution map, which are sized (N,K) for an N batch of a K-multinomial
        """
        return (lambda dist: (1 - t)*dist + (t)*torch.ones_like(dist)/dist.shape[1])

    def create_teams(self,
                     T,
                     N=1,
                     obs_preembed=None,
                     obs_mask=None,
                     noise_model=None,
                     ):

        """
        creates random teams of size (N,T)
        Args:
            T: number of team members
            N: number of teams to make (default 1)
            obs_preembed: input to give input embedder, size (N,S,*)
                None if no input
            obs_mask: size (N,S) boolean array of whether to mask each input embedding
            noise_model: noise to add to probability distribution, if None, doesnt add noise
        Returns:
            filled in teams of size (N,T)
        """
        return self.fill_in_teams(initial_teams=self.create_masked_teams(T=T, N=N),
                                  obs_preembed=obs_preembed,
                                  obs_mask=obs_mask,
                                  noise_model=noise_model,
                                  )

    def fill_in_teams(self,
                      initial_teams,
                      obs_preembed=None,
                      obs_mask=None,
                      noise_model=None,
                      num_masks=None,
                      ):
        """
        replaces all [MASK] with team members
            repeatedly calls mutate_add_member
        Args:
            initial_teams: initial torch array of team members, shape (N,T)
            obs_preembed: input to give input embedder, size (N,S,*)
                None if no input
            obs_mask: size (N,S) boolean array of whether to mask each input embedding
            noise_model: noise to add to probability distribution, if None, doesnt add noise
            num_masks: specify the largest number of [MASK] tokens in any row
                if None, just calls mutate_add_member (T) times
        Returns:
            filled in teams of size (N,T)
        """
        N, T = initial_teams.shape
        if num_masks is None:
            num_masks = T
        for _ in range(num_masks):
            initial_teams = self.mutate_add_member(initial_teams=initial_teams,
                                                   indices=None,
                                                   obs_preembed=obs_preembed,
                                                   obs_mask=obs_mask,
                                                   noise_model=noise_model
                                                   )
        return initial_teams

    def mutate_add_member(self,
                          initial_teams,
                          indices=None,
                          obs_preembed=None,
                          obs_mask=None,
                          noise_model=None,
                          ):
        """
        updates initial teams by updating specified indices with samples from the probability distribution
            if indices is None, chooses indices that correspond with [MASK] tokens (one for each row)
        Args:
            initial_teams: initial torch array of team members, shape (N,T)
            indices: indices to replace (if None, picks one masked index at random)
                should be in pytorch format, (list of dim 0 indices, list of dim 1 indices)
            obs_preembed: input to give input embedder, size (N,S,*)
                None if no input
            obs_mask: size (N,S) boolean array of whether to mask each input embedding
            noise_model: noise to add to probability distribution, if None, doesnt add noise
        Returns:
            team with updates
        """
        N, T = initial_teams.shape
        if indices is None:
            indices = [[], []]
            for i in range(N):
                potential = torch.where(initial_teams[i] == self.MASK)[0]
                if len(potential) > 0:
                    indices[0].append(i)
                    indices[1].append(potential[torch.randint(0, len(potential), (1,))])
                # else:
                #    print("warning: tried to add to full team")
        if len(indices[0]) == 0:
            # no [MASK] tokens exist
            return initial_teams
        output = self.team_builder.forward(obs_preembed=obs_preembed,
                                           target_team=initial_teams,
                                           obs_mask=obs_mask,
                                           output_probs=True,
                                           pre_softmax=False,
                                           )
        dist = output[indices]  # (|indices|,num_agents) multinomial distribution for each index to update
        if noise_model is not None:
            # add noise if this is a thing
            dist = noise_model(dist)
        # torch.multinomial samples each

        initial_teams[indices] = torch.multinomial(dist, 1).flatten()
        return initial_teams


class DiscreteInputTrainer(TeamTrainer):
    def __init__(self,
                 num_agents,
                 num_input_tokens,
                 embedding_dim=512,
                 nhead=8,
                 num_encoder_layers=16,
                 num_decoder_layers=16,
                 dim_feedforward=None,
                 dropout=.1,
                 pos_encode_input=True,
                 append_pos_encode=False,
                 ):
        berteam = BERTeam(num_agents=num_agents,
                          embedding_dim=embedding_dim,
                          nhead=nhead,
                          num_encoder_layers=num_encoder_layers,
                          num_decoder_layers=num_decoder_layers,
                          dim_feedforward=dim_feedforward,
                          dropout=dropout,
                          )
        if pos_encode_input:
            if append_pos_encode:
                Constructor = DiscreteInputPosAppender
            else:
                Constructor = DiscreteInputPosEmbedder
            input_embedder = Constructor(num_embeddings=num_input_tokens,
                                         embedding_dim=embedding_dim,
                                         dropout=dropout,
                                         )
        else:
            input_embedder = DiscreteInputEmbedder(num_embeddings=num_input_tokens,
                                                   embedding_dim=embedding_dim,
                                                   )

        super().__init__(
            team_builder=TeamBuilder(
                berteam=berteam,
                input_embedder=input_embedder,
            )
        )


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    N = 32
    S = 1
    T = 1
    epochs = 100
    torch.random.manual_seed(69)

    E = 64

    num_inputs = 3
    test = DiscreteInputTrainer(num_agents=69,
                                num_input_tokens=num_inputs,
                                embedding_dim=E,
                                pos_encode_input=True,
                                append_pos_encode=False,
                                )

    init_team = torch.arange(T, dtype=torch.long)


    def random_shuffle():
        return init_team[torch.randperm(len(init_team))]


    losses = []
    for epoch in range(epochs):
        # input_preembedding_gen = (torch.randint(0, num_inputs, (i%S,)) if i > 0 else None for i in range(N))
        # input_embedding_gen = (None for i in range(N))

        input_mask = [torch.nan for _ in range(N)]
        input_preembedding = torch.randint(0, num_inputs, (N, S))
        # input_mask = torch.zeros((N, S), dtype=torch.bool)
        # out_teams = torch.stack([random_shuffle() for _ in range(N)], dim=0)
        out_teams = []
        for i, input_pre in enumerate(input_preembedding):
            out_teams.append(torch.ones(T, dtype=torch.long)*(input_pre[0] + 1)%3)

        out_teams = torch.stack(out_teams, dim=0)

        data = list(zip(input_preembedding, out_teams, input_mask))
        loader = DataLoader(data,
                            shuffle=True,
                            batch_size=64,
                            )
        loss = test.training_step(
            loader=loader,
            minibatch=True
        )

        losses.append(loss)
        print('epoch', epoch, '\tloss', loss.item())

    # print(test.mask_and_learn(input_embedding_gen=input_embeddings,
    #                          winning_team_gen=(init_team.clone() for _ in range(N))))
    num_teams = 9
    init_teams = test.create_masked_teams(T=T, N=num_teams)
    print(init_teams)
    input_pre = torch.arange(0, num_teams).view((-1, 1))%num_inputs
    for _ in range(T):
        test.mutate_add_member(initial_teams=init_teams, obs_preembed=input_pre)
        print(init_teams)
    print('target:')
    print(input_pre)
    plt.plot(range(10, len(losses)), losses[10:])
    plt.show()
