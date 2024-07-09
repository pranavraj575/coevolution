import torch
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


def single_game_outcome(i, j, agents):
    diff = (agents[i] - agents[j])%3
    if diff == 0:  # the agents tied
        return 0
    # otherwise, 1 if agent i won, and 2 if agent j won
    return 3 - 2*diff  # need to return 1 if agent i won and -1 if agent j won


if __name__ == '__main__':
    torch.random.manual_seed(69)
    print(outcomes((torch.tensor([[0], [1], [2], [0]]), torch.tensor(([[1], [0], [0], [0]]))))[-1])
