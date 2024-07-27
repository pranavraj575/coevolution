import torch
from matplotlib import pyplot as plt
from src.game_outcome import PlayerInfo, OutcomeFn, PettingZooOutcomeFn

ROCK = 0
PAPER = 1
SCISOR = 2


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


class SingleOutcome(OutcomeFn):
    def __init__(self, agents):
        self.agents = agents

    def get_outcome(self, team_choices, train_infos=None, env=None):
        i, j = team_choices
        diff = (self.agents[i[0]] - self.agents[j[0]])%3
        if diff == 0:  # the agents tied
            choice = self.agents[i,]
            return [(.5, [PlayerInfo(obs_preembed=choice)]),
                    (.5, [PlayerInfo(obs_preembed=choice)])]
        if diff == 1:
            # agent i won
            return [
                (1, [PlayerInfo(obs_preembed=self.agents[j,])]),
                (0, [PlayerInfo(obs_preembed=self.agents[i,])]),
            ]

        if diff == 2:
            # agent j won
            return [
                (0, [PlayerInfo(obs_preembed=self.agents[j,])]),
                (1, [PlayerInfo(obs_preembed=self.agents[i,])]),
            ]


class SingleZooOutcome(PettingZooOutcomeFn):

    def _get_outcome_from_agents(self, agent_choices, index_choices, train_infos, env):
        a, b = agent_choices
        a = a[0]
        b = b[0]
        diff = (a - b)%3
        if diff == 0:  # the agents tied
            return [(.5, [PlayerInfo(obs_preembed=torch.tensor(b).view((-1, 1)))]),
                    (.5, [PlayerInfo(obs_preembed=torch.tensor(a).view((-1, 1)))])]
        if diff == 1:
            # agent i won
            return [
                (1, [PlayerInfo(obs_preembed=torch.tensor(b).view((-1, 1)))]),
                (0, [PlayerInfo(obs_preembed=torch.tensor(a).view((-1, 1)))]),
            ]

        if diff == 2:
            # agent j won
            return [
                (0, [PlayerInfo(obs_preembed=torch.tensor(b).view((-1, 1)))]),
                (1, [PlayerInfo(obs_preembed=torch.tensor(a).view((-1, 1)))]),
            ]


if __name__ == '__main__':
    torch.random.manual_seed(69)
    agents = torch.arange(3)
    outcomes = SingleOutcome(agents=agents)
    print(outcomes.get_outcome(team_choices=(torch.tensor([0]), torch.tensor([1]))))
    print(outcomes.get_outcome(team_choices=(torch.tensor([1]), torch.tensor([1]))))
    print(outcomes.get_outcome(team_choices=(torch.tensor([2]), torch.tensor([1]))))
