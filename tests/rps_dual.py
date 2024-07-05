import torch

ROCK = 0
PAPER = 1
SCISOR = 2


def tie_games(playersA, playersB):
    """
    Returns the tying agents, as well as the opponent
    Args:
        playersA: (N,2) array of ROCK, PAPER, or SCISORS
        playersB: (N,2) array of ROCK, PAPER, or SCISORS
    Returns:
        tyers: (N',2) array of tying teams
        observations: (N',2) array of opponent tying teams
        indices: which games had tyers, in the order returned
    """
    N, _ = playersA.shape
    indices = []
    for i in range(N):
        if torch.all(playersA[i] == playersB[i]) or torch.all(playersA[i].__reversed__() == playersB[i]):
            indices.append(i)
    return playersA[indices, :], playersB[indices, :], indices


def winners(playersA, playersB):
    """
    Returns the winning agents, as well as the opponent
    Args:
        playersA: (N,2) array of ROCK, PAPER, or SCISORS
        playersB: (N,2) array of ROCK, PAPER, or SCISORS
    Returns:
        winners: (N',2) array of winning teams
        observations: (N',2) array of winning teams
        indices: which games had winners, in the order returned
    """
    N, _ = playersA.shape
    indices = []
    for i in range(N):
        if not (torch.all(playersA[i] == playersB[i]) or
                torch.all(playersA[i].__reversed__() == playersB[i])):
            indices.append(i)
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
    return winners, losers, indices


if __name__ == '__main__':
    torch.random.manual_seed(69)
    N = 3
    # playersA = torch.randint(0, 3, (N, 2))
    # playersB = torch.randint(0, 3, (N, 2))
    playersA = torch.tensor([[0, 0]])
    playersB = torch.tensor([[0, 1]])
    print(playersA)
    print(playersB)
    print(winners(playersA, playersB))
