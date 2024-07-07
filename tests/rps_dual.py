import torch
from src.team_trainer import DiscreteInputTrainer
from src.replay_buffer_dataset import ReplayBufferDiskStorage
from torch.utils.data import DataLoader
import numpy as np
import itertools

ROCK = 0
PAPER = 1
SCISOR = 2


def outcomes(playersA, playersB):
    """
    Returns the winning agents, as well as the opponent
    Args:
        playersA: (N,2) array of ROCK, PAPER, or SCISORS
        playersB: (N,2) array of ROCK, PAPER, or SCISORS
    Returns:
        (
        winners: (N',2) array of winning teams
        observations: (N',2) array of losing teams
        indices: which games had winners, in the order returned
        ),
        (
        tiedA: (N'',2) array of playrs from A that tied
        tiedB: (N'',2) array of playrs from B that tied
        tied_games: indices that tied
        )
    """
    N, _ = playersA.shape
    indices = []
    tied_games = []
    for i in range(N):
        if (torch.all(playersA[i] == playersB[i]) or torch.all(playersA[i].__reversed__() == playersB[i])):
            tied_games.append(i)
        else:
            indices.append(i)

    tiedA, tiedB = playersA[tied_games], playersB[tied_games]

    playersA = playersA[indices, :]
    playersB = playersB[indices, :]
    N_p = len(indices)
    deletedA = torch.zeros((N_p, 2))
    deletedB = torch.zeros((N_p, 2))
    for delar, player, opponent in ((deletedA, playersA, playersB), (deletedB, playersB, playersA)):
        for i in range(N_p):
            for k in range(2):
                deletor = (player[i, k] + 1)%3
                delar[i, k] = deletor in opponent[i]
    fails_A = torch.sum(deletedA, dim=1)
    fails_B = torch.sum(deletedB, dim=1)

    # 1 where B wins, 0 otherwise
    B_wins = (torch.sign(fails_A - fails_B)/2 + .5).long()

    both = torch.stack((playersA, playersB), dim=0)
    # both[0,:,:] is playersA, and both[:,:,:] is B
    winners = both[B_wins, torch.arange(N_p), :]
    losers = both[1 - B_wins, torch.arange(N_p), :]
    return (winners, losers, indices), (tiedA, tiedB, tied_games)


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
                                   pos_encode_input=False,
                                   num_decoder_layers=4,
                                   num_encoder_layers=4,
                                   )

    N = 1000
    capacity = int(1e5)
    buffer = ReplayBufferDiskStorage(storage_dir=os.path.join(DIR, "data", "temp", "tests_rps2"), capacity=capacity)

    minibatch = 64
    init_dists = []
    cond_dists = []
    strat_labels = ['RR', 'PP', 'SS', 'RP', 'RS', 'PS']
    losses = []
    epoch = 0
    steps_each_time = 2
    for _ in range(100):
        start_time = time.time()
        noise = trainer.create_nose_model_towards_uniform(1/np.sqrt(epoch + 1))
        players, opponents = (trainer.create_teams(T=2, N=N, noise_model=noise),
                              trainer.create_teams(T=2, N=N, noise_model=noise))
        (winners, losers, indices), _ = outcomes(players, opponents)

        against_winners = trainer.create_teams(T=2,
                                               N=len(winners),
                                               obs_preembed=winners.view((-1, 2)),
                                               noise_model=noise,
                                               )
        (winners2, losers2, indices2), _ = outcomes(winners.view((-1, 2)), against_winners)
        for a, b in [(winners, losers),
                     (winners2, losers2),
                     ]:
            for c in [
                list(torch.tensor([True, True]) for _ in b),
                list(torch.tensor([True, False]) for _ in b),
                list(torch.tensor([False, True]) for _ in b),
                list(torch.tensor([False, False]) for _ in b)]:
                buffer.extend(zip(a, b, c))
        game_time = time.time() - start_time

        for step in range(steps_each_time):
            start_time = time.time()
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
            # data is in (context, winning team, mask)
            # N=1 T=2
            # dataloaders want shape of (T,)

            masked = lambda loser: (not torch.is_tensor(loser)) or torch.all(torch.isnan(loser))

            data = (((loser.view(2),
                      winner.view(2),
                      mask,
                      )
                     for winner, loser, mask in sample))
            loader = DataLoader(list(data), shuffle=True, batch_size=16)
            loss = trainer.training_step(loader=loader, minibatch=False).item()
            losses.append(loss)

            print('epoch', epoch, ';\tbuffer size', len(buffer))
            print('loss', loss)
            if step == 0:
                print('\tgame_time:', round(game_time, 2))
            print('\ttrain time:', round(time.time() - start_time, 2))

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
