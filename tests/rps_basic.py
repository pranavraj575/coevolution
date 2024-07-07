import torch
from src.team_trainer import DiscreteInputTrainer
from src.replay_buffer_dataset import ReplayBufferDiskStorage
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt

ROCK = 0
PAPER = 1
SCISOR = 2


def outcomes(teams):
    """
    Returns the winning agents, as well as the contexts
    Args:
        teams: 2-tuple of (N, 1) arrays of players
    Returns:
        (
            win_indices: which games had winners (game number list, indices of winning team), both size (N')
            pre_embedded_observations: (N', 1) array observed by winning team
            embedding_mask: None
        ),
        (
            loss_indices: which games had losers (game number list, indices of winning team), both size (N'')
            pre_embedded_observations: (N'', 1) array observed by losing team
            embedding_mask: None
        ),
        (
            tied_indices: indices that tied, (game number list, indices of tied team), both size (N''')
            pre_embedded_observations: (N''', 1) array observed by tied team
            embedding_mask: None
        )
    """
    A, B = teams
    wl_games = torch.where(A != B)[0]
    tied_games = torch.where(A == B)[0]
    tiedA, tiedB = A[tied_games], B[tied_games]

    A = A[wl_games]
    B = B[wl_games]

    diffs = (A - B)
    # if 1 or -2, A won
    # if -1 or 2, B won
    win_teams = (diffs%3).flatten() - 1  # if 1, A won, if 2, B won

    both = torch.cat((A, B), dim=1)  # first column is A, second is B
    N_p, _ = both.shape
    win_obs = both[torch.arange(N_p), 1 - win_teams].view((-1, 1))
    loss_obs = both[torch.arange(N_p), win_teams].view((-1, 1))

    win_indices = (wl_games, win_teams)
    loss_indices = (wl_games, 1 - win_teams)
    tied_indices = (torch.cat((tied_games, tied_games)),
                    torch.cat((torch.zeros_like(tied_games), torch.ones_like(tied_games)))
                    )
    tied_obs = torch.cat((tiedB.view((-1, 1)), tiedA.view((-1, 1))), dim=0)
    return (
        (win_indices, win_obs, None),
        (loss_indices, loss_obs, None),
        (tied_indices, tied_obs, None),
    )


def get_winners_and_losers(teams, outcomes):
    """
    for two team uniform team size games, gets winner teams and loser teams
    Args:
        teams:
        outcomes:
    Returns:
    """
    ((indices, win_idxs), _, _), ((indices, lose_idxs), _, _), _ = outcomes(teams)
    both = torch.stack(teams, dim=0)
    return both[win_idxs, indices], both[lose_idxs, indices]


def loss_plot(losses, save_dir=None, show=False):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.title('CrossEntropy Loss')
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_dist_evolution(plot_dist, save_dir=None, show=False, alpha=1, labels='RPS', title=None):
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
    plt.legend(loc='center left', bbox_to_anchor=(-.25, .5))

    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    import os, sys

    torch.random.manual_seed(69)

    # print(outcomes((torch.tensor([[0], [1], [2], [0]]), torch.tensor(([[1], [0], [0], [0]]))))[-1])

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
    plot_dir = os.path.join(DIR, 'data', 'plots', 'tests_rps')
    if not os.path.exists((plot_dir)):
        os.makedirs(plot_dir)
    trainer = DiscreteInputTrainer(num_agents=3,
                                   num_input_tokens=3,
                                   embedding_dim=64,
                                   pos_encode_input=False,
                                   num_decoder_layers=4,
                                   num_encoder_layers=4,
                                   )
    N = 100
    capacity = int(1e5)
    buffer = ReplayBufferDiskStorage(storage_dir=os.path.join(DIR, "data", "temp", "tests_rps"), capacity=capacity)

    minibatch = 64
    init_dists = []
    cond_dists = []
    losses = []
    for epoch in range(100):
        noise = trainer.create_nose_model_towards_uniform(.1)
        for preembed, team, mask in trainer.collect_training_data(outcome=outcomes,
                                                                  num_members=(1, 1),
                                                                  N=N,
                                                                  number_of_tie_matches=0,
                                                                  number_of_loss_rematches=1,
                                                                  noise_model=noise
                                                                  ):
            buffer.push((preembed, team.unsqueeze(0), mask))

            # buffer.push((None, team.unsqueeze(0), None))
        init_distribution = trainer.team_builder.forward(obs_preembed=None,
                                                         target_team=trainer.create_masked_teams(T=1, N=1),
                                                         obs_mask=None,
                                                         ).detach().flatten().numpy()
        init_dists.append(init_distribution)
        conditional_dists = []
        for opponent in range(3):
            dist = trainer.team_builder.forward(obs_preembed=torch.tensor([[opponent]]),
                                                target_team=trainer.create_masked_teams(T=1, N=1),
                                                obs_mask=None,
                                                )
            conditional_dists.append(dist.detach().flatten().numpy())
        cond_dists.append(conditional_dists)
        sample = buffer.sample(batch=minibatch)

        loss = trainer.training_step(data=sample, minibatch=False).item()
        losses.append(loss)
        print('epoch', epoch, ';\tbuffer size', len(buffer))
        print('loss', loss)
        if not (epoch + 1)%10:
            loss_plot(losses=losses, save_dir=os.path.join(plot_dir, 'loss_plot.png'))
            plot_dist_evolution(init_dists, save_dir=os.path.join(plot_dir, 'init_dist.png'))
            for k, name in enumerate('RPS'):
                plot_dist_evolution([dist[k] for dist in cond_dists],
                                    save_dir=os.path.join(plot_dir, 'dist_against' + name + '.png'),
                                    title='Distribution against ' + name)
    buffer.close()
