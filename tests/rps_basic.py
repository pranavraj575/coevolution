import torch

ROCK = 0
PAPER = 1
SCISOR = 2


def tie_games(playersA, playersB):
    """
    Returns the tying agents, as well as the opponent
    Args:
        playersA: (N,1) array of either ROCK, PAPER, or SCISORS
        playersB: (N,1) array of either ROCK, PAPER, or SCISORS
    Returns:
        tyers: (N',1) array of either ROCK, PAPER, or SCISORS that the player played
        observations: (N',1) array of either ROCK, PAPER, or SCISORS that the opponent played
        indices: which games had winners, in the order returned
    """
    indices = torch.where(playersA == playersB)[0]
    return playersA[indices], playersB[indices], indices


def winners(playersA, playersB):
    """
    Returns the winning agents, as well as the opponent
    Args:
        playersA: (N,1) array of either ROCK, PAPER, or SCISORS
        playersB: (N,1) array of either ROCK, PAPER, or SCISORS

    Returns:
        winners: (N',1) array of either ROCK, PAPER, or SCISORS that the player played
        observations: (N',1) array of either ROCK, PAPER, or SCISORS that the opponent played
        indices: which games had winners, in the order returned
    """
    indices = torch.where(playersA != playersB)[0]
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

    return winners, losers, indices


if __name__ == '__main__':
    print(winners(torch.tensor([[0], [1], [2], [0]]), torch.tensor(([[1], [0], [0], [0]]))))
