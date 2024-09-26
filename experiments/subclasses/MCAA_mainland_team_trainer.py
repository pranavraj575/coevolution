import torch
import numpy as np

from BERTeam.trainer import TeamTrainer


class MCAAMainland(TeamTrainer):
    def __init__(self, pop_sizes, MASK=-1, softmax_update=32*np.log(10)/400):
        num_agents = sum(pop_sizes)
        super().__init__(num_agents, MASK=MASK)
        self.pop_sizes = pop_sizes
        self.num_islands = len(pop_sizes)
        self.cum_islands = torch.cumsum(torch.tensor(pop_sizes), dim=0)
        self.distribution = torch.zeros(self.num_islands)
        self.distribution = torch.arange(self.num_islands, dtype=torch.float)
        # this will be a softmax, and will be updated though gradient backprop
        self.buffer = torch.zeros(self.num_islands)
        self.softmax_update = softmax_update

    def global_indices_to_islands(self, team):
        """
        returns which island each index in an array is on
        Args:
            team: vector of population indices
        Returns:
            vector of island indices
        """
        return torch.sum(team.view(-1, 1) > self.cum_islands, dim=1)

    def train(self, *args, **kwargs):
        # TODO: update island distribution with team
        target = self.buffer/torch.sum(self.buffer)

        # jacobian of softmax, let output of softmax be (s_i)_i
        # diagonal entries are s_i(1-s_i)
        # off diagonal entry is -s_is_j
        soft_dist = torch.softmax(self.distribution, dim=-1)
        softmax_jacob = torch.diag(torch.square(soft_dist) + soft_dist*(1 - soft_dist)) - torch.matmul(
            soft_dist.view(-1, 1),
            soft_dist.view(1, -1)
        )
        # error = target - 1*torch.log(soft_dist)
        error = target - soft_dist
        update = torch.matmul(error.view(1, -1), softmax_jacob)

        self.distribution += self.softmax_update*update.flatten()

        # reset buffer
        self.buffer = torch.zeros(self.num_islands)

    def add_to_buffer(self, scalar, obs_preembed, team, obs_mask):
        # TODO: update buffer with team rank
        #  if scalar is some number
        if scalar == 1:
            islands = self.global_indices_to_islands(team=team)
            for island_idx in islands:
                self.buffer[island_idx] += 1

    def fill_in_teams(self,
                      initial_teams,
                      noise_model=None,
                      num_masks=None,
                      valid_members=None,
                      **kwargs
                      ):
        # TODO: fill in remaining members according to distribution
        indices = torch.where(torch.eq(initial_teams, self.MASK))
        fill = torch.multinomial(torch.softmax(self.distribution, dim=-1),
                                 num_samples=len(indices[0]),
                                 replacement=True,
                                 )
        initial_teams[indices] = fill
        return initial_teams

    def create_teams(self,
                     T,
                     N=1,
                     obs_preembed=None,
                     obs_mask=None,
                     noise_model=None,
                     ):
        return torch.multinomial(torch.softmax(self.distribution, dim=-1),
                                 num_samples=T*N,
                                 replacement=True,
                                 ).reshape((N, T))


if __name__ == '__main__':
    thing = MCAAMainland(pop_sizes=(1, 3, 3, 4), softmax_update=1)
    goal_team = torch.tensor([0, 1, 2, 3, 4])
    goal_dist = thing.global_indices_to_islands(goal_team)
    print(goal_dist)
    print(torch.softmax(thing.distribution, dim=-1))
    for i in range(1000):
        thing.add_to_buffer(1, None, goal_team, None)
        thing.train()
    print(torch.softmax(thing.distribution, dim=-1))
    print(thing.distribution)
    print(thing.create_teams(10, N=4))
    fake_team = thing.create_masked_teams(10)
    fake_team[0, :5] = 69
    print(thing.fill_in_teams(initial_teams=fake_team))
