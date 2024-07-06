import torch
from src.team_trainer import DiscreteInputTrainer
from src.replay_buffer_dataset import ReplayBufferDiskStorage
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt

ROCK = 0
PAPER = 1
SCISOR = 2


def outcomes(playersA, playersB):
    """
    Returns the winning agents, as well as the opponent
    Args:
        playersA: (N,1) array of either ROCK, PAPER, or SCISORS
        playersB: (N,1) array of either ROCK, PAPER, or SCISORS

    Returns:
        (
        winners: (N',1) array of either ROCK, PAPER, or SCISORS that the player played
        observations: (N',1) array of either ROCK, PAPER, or SCISORS that the opponent played
        indices: which games had winners, in the order returned
        ),
        (
        tiedA: (N'',1) array of playrs from A that tied
        tiedB: (N'',1) array of playrs from B that tied
        tied_indices: indices that tied
        )
    """
    indices = torch.where(playersA != playersB)[0]
    tied_games = torch.where(playersA == playersB)[0]
    tiedA, tiedB = playersA[tied_games], playersB[tied_games]

    playersA = playersA[indices]
    playersB = playersB[indices]

    diffs = (playersA - playersB)
    # if 1 or -2, player A won
    # if -1 or 2, player B won
    diffs = (diffs%3).flatten()  # if 1, player A won, if 2, player B won
    both = torch.cat((playersA, playersB), dim=1)  # first column is A, second is B
    N_p, _ = both.shape
    winners = both[torch.arange(N_p), diffs - 1]
    losers = both[torch.arange(N_p), 2 - diffs]

    return (winners, losers, indices), (tiedA, tiedB, tied_games)


def plot_dist_evolution(plot_dist, save_dir=None, show=False, alpha=.5, labels='RPS', title=None):
    num_pops = len(plot_dist[0])
    x = range(len(plot_dist))
    for i in range(num_pops):
        plt.fill_between(x=x,
                         y1=[sum(dist[:i]) for dist in plot_dist],
                         y2=[sum(dist[:i + 1]) for dist in plot_dist],
                         label=labels[i] if labels is not None and len(labels) >= i else None,
                         alpha=alpha,
                         )
    if title is not None:
        plt.title(title)
    plt.legend()

    if save_dir is not None:
        plt.savefig(save_dir)
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    import os, sys

    torch.random.manual_seed(69)

    outcomes(torch.tensor([[0], [1], [2], [0]]), torch.tensor(([[1], [0], [0], [0]])))

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
    plot_dir = os.path.join(DIR, 'data', 'plots', 'tests')
    if not os.path.exists((plot_dir)):
        os.makedirs(plot_dir)
    trainer = DiscreteInputTrainer(num_agents=3,
                                   num_input_tokens=3,
                                   )

    N = 100
    capacity = int(1e5)
    buffer = ReplayBufferDiskStorage(storage_dir=os.path.join(DIR, "data", "temp"), capacity=capacity)

    minibatch = 64
    init_dists = []
    cond_dists = []
    for i in range(1000):
        noise = trainer.create_nose_model_towards_uniform(.05)
        players, opponents = (trainer.create_teams(T=1, N=100, noise_model=noise),
                              trainer.create_teams(T=1, N=100, noise_model=noise))
        (winners, losers, indices), _ = outcomes(players, opponents)
        buffer.extend(zip(winners, losers))
        # buffer.extend(zip(winners, (torch.nan for _ in losers)))
        init_distribution = trainer.team_builder.forward(input_preembedding=None,
                                                         target_team=trainer.create_masked_teams(T=1, N=1),
                                                         input_mask=None,
                                                         ).detach().flatten().numpy()
        init_dists.append(init_distribution)
        conditional_dists = []
        for opponent in range(3):
            dist = trainer.team_builder.forward(input_preembedding=torch.tensor([[opponent]]),
                                                target_team=trainer.create_masked_teams(T=1, N=1),
                                                input_mask=None,
                                                )
            conditional_dists.append(dist.detach().flatten().numpy())
        cond_dists.append(conditional_dists)
        print('distributions [R,P,S]')
        print('initial distribution', np.round(init_distribution, 2))
        for k, name in enumerate('RPS'):
            print('\tdistribution against', name + ':', np.round(conditional_dists[k], 2))
        print()
        print('buffer size', len(buffer))
        sample = buffer.sample(batch=minibatch)
        # data is in (context, winning team, mask)
        # N=1 T=1
        # dataloaders want shape of (T,)

        masked = lambda loser: (not torch.is_tensor(loser)) or torch.all(torch.isnan(loser))

        data = (((torch.tensor([0]) if masked(loser) else loser.view(1),
                  winner.view(1),
                  torch.tensor([True]) if masked(loser) else torch.tensor([False]),
                  )
                 for winner, loser in sample))
        loader = DataLoader(list(data), shuffle=True, batch_size=16)
        trainer.epoch(loader=loader, minibatch=False)
        if not (i + 1)%10:
            plot_dist_evolution(init_dists, save_dir=os.path.join(plot_dir, 'init_dist.png'))
            for k, name in enumerate('RPS'):
                plot_dist_evolution([dist[k] for dist in cond_dists],
                                    save_dir=os.path.join(plot_dir, 'dist_against' + name + '.png'),
                                    title='Distribution against ' + name)
