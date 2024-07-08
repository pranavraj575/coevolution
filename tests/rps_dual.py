import torch
from src.team_trainer import DiscreteInputTrainer
from src.replay_buffer_dataset import ReplayBufferDiskStorage
import numpy as np
import itertools

ROCK = 0
PAPER = 1
SCISOR = 2


def outcomes(teams):
    """
    Returns the winning agents, as well as the contexts
    Args:
        teams: 2-tuple of (N, 2) arrays of players
    Returns:
        (
            win_indices: which games had winners (game number list, indices of winning team), both size (N')
            pre_embedded_observations: (N', 2) array observed by winning team
            embedding_mask: None
        ),
        (
            loss_indices: which games had losers (game number list, indices of winning team), both size (N'')
            pre_embedded_observations: (N'', 2) array observed by losing team
            embedding_mask: None
        ),
        (
            tied_indices: indices that tied, (game number list, indices of tied team), both size (N''')
            pre_embedded_observations: (N''', 2) array observed by tied team
            embedding_mask: None
        )
    """
    A, B = teams
    N, _ = A.shape
    wl_games = []
    tied_games = []
    for i in range(N):
        if (torch.all(A[i] == B[i]) or torch.all(A[i].__reversed__() == B[i])):
            tied_games.append(i)
        else:
            wl_games.append(i)
    tied_games = torch.tensor(tied_games, dtype=torch.long)

    tiedA, tiedB = A[tied_games], B[tied_games]

    A = A[wl_games, :]
    B = B[wl_games, :]
    N_p = len(wl_games)
    deletedA = torch.zeros((N_p, 2))
    deletedB = torch.zeros((N_p, 2))
    for delar, player, opponent in ((deletedA, A, B), (deletedB, B, A)):
        for i in range(N_p):
            for k in range(2):
                deletor = (player[i, k] + 1)%3
                delar[i, k] = deletor in opponent[i]
    fails_A = torch.sum(deletedA, dim=1)
    fails_B = torch.sum(deletedB, dim=1)

    # 1 where B wins, 0 otherwise
    win_teams = (torch.sign(fails_A - fails_B)/2 + .5).long()

    both = torch.stack((A, B), dim=0)
    # both[0,:,:] is A, and both[1,:,:] is B

    win_indices = (wl_games, win_teams)
    loss_indices = (wl_games, 1 - win_teams)
    tied_indices = (torch.cat((tied_games, tied_games)),
                    torch.cat((torch.zeros_like(tied_games), torch.ones_like(tied_games)))
                    )

    tied_obs = torch.cat((tiedB.view((-1, 2)), tiedA.view((-1, 2))), dim=0)
    win_obs = both[1 - win_teams, torch.arange(N_p), :].view((-1, 2))

    loss_obs = both[win_teams, torch.arange(N_p), :].view((-1, 2))

    return (
        (win_indices, win_obs, None),
        (loss_indices, loss_obs, None),
        (tied_indices, tied_obs, None),
    )
    return (winners, losers, wl_games), (tiedA, tiedB, tied_games)


def dist_from_trainer(trainer: DiscreteInputTrainer,
                      input_preembedding=None,
                      input_mask=None,
                      num_agents=3,
                      keyorder=None
                      ):
    trainer.team_builder.eval()
    dic = dict()
    keys = []
    for choice in itertools.chain(itertools.combinations(range(num_agents), 1),
                                  itertools.combinations(range(num_agents), 2)):
        if len(choice) == 1:
            choice = choice + choice
            perms = [choice]
        else:
            perms = itertools.permutations(choice)
        keys.append(choice)
        prob = 0
        for perm in perms:
            orders = list(itertools.permutations(range(2)))
            for order in orders:
                this_prob = 1.
                target_team = trainer.create_masked_teams(T=2, N=1)

                for idx in order:
                    full_dist = trainer.team_builder.forward(input_preembedding, target_team, input_mask)
                    dist = full_dist[0, idx]
                    this_prob *= dist[perm[idx]].item()
                    target_team[0, idx] = perm[idx]
                prob += this_prob/len(orders)
        dic[choice] = prob

    trainer.team_builder.train()
    if keyorder is None:
        keyorder = keys
    return np.array([dic[key] for key in keyorder])


if __name__ == '__main__':
    import os, sys
    from tests.rps_basic import plot_dist_evolution, loss_plot
    import time

    torch.random.manual_seed(69)

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
    plot_dir = os.path.join(DIR, 'data', 'plots', 'tests_rps2')
    if not os.path.exists((plot_dir)):
        os.makedirs(plot_dir)
    trainer = DiscreteInputTrainer(num_agents=3,
                                   num_input_tokens=3,
                                   embedding_dim=64,
                                   pos_encode_input=True,
                                   append_pos_encode_input=True,
                                   pos_encode_teams=True,
                                   append_pos_encode_teams=True,
                                   num_decoder_layers=4,
                                   num_encoder_layers=4,
                                   )

    N = 100
    capacity = int(2e4)
    buffer = ReplayBufferDiskStorage(storage_dir=os.path.join(DIR, "data", "temp", "tests_rps2"), capacity=capacity)

    minibatch = 64
    init_dists = []
    cond_dists = []
    strat_labels = ['RR', 'PP', 'SS', 'RP', 'RS', 'PS']
    losses = []
    for epoch in range(100):
        start_time = time.time()
        noise = trainer.create_nose_model_towards_uniform(1/np.sqrt(epoch/2 + 1))
        for scalar, preembed, team, mask in trainer.collect_training_data(outcome=outcomes,
                                                                          num_members=(2, 2),
                                                                          N=N,
                                                                          number_of_tie_matches=0,
                                                                          number_of_loss_rematches=3,
                                                                          noise_model=noise,
                                                                          obtain_negatives=False,
                                                                          ):
            buffer.push((scalar, preembed, team, mask))
            buffer.push((scalar, None, team, None))

        init_distribution = dist_from_trainer(trainer=trainer,
                                              input_preembedding=None,
                                              input_mask=None,
                                              )
        init_dists.append(init_distribution)
        conditional_dists = []
        for opponent_choice in itertools.chain(itertools.combinations(range(3), 1),
                                               itertools.combinations(range(3), 2)):
            if len(opponent_choice) == 1:
                opponent_choice = opponent_choice + opponent_choice
            dist = dist_from_trainer(trainer=trainer,
                                     input_preembedding=torch.tensor([opponent_choice]),
                                     input_mask=None,
                                     )
            conditional_dists.append(dist)
        cond_dists.append(conditional_dists)

        sample = buffer.sample(batch=minibatch)
        collection_time=time.time() - start_time
        start_time = time.time()
        loss = trainer.training_step(data=sample, minibatch=False).item()
        losses.append(loss)

        print('epoch', epoch, ';\tbuffer size', len(buffer))
        print('\tcollection time:', round(collection_time, 2))
        print('\ttrain time:', round(time.time() - start_time, 2))

        print('\tloss', loss)

        if not (epoch + 1)%10:
            start_time = time.time()
            loss_plot(losses, save_dir=os.path.join(plot_dir, 'loss_plot.png'))

            plot_dist_evolution(init_dists,
                                save_dir=os.path.join(plot_dir, 'init_dist.png'),
                                labels=strat_labels
                                )
            for k, name in enumerate(strat_labels):
                plot_dist_evolution([dist[k] for dist in cond_dists],
                                    save_dir=os.path.join(plot_dir, 'dist_against' + name + '.png'),
                                    title='Distribution against ' + name,
                                    labels=strat_labels,
                                    )
            print('\tplot time:', round(time.time() - start_time, 2))
        epoch += 1
        print()
    buffer.close()
