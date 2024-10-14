import torch, os, pickle, itertools
import numpy as np

from BERTeam.trainer import TeamTrainer


class MCAAMainland(TeamTrainer):
    def __init__(self, pop_sizes, MASK=-1, softmax_update=32*np.log(10)/400):
        num_agents = sum(pop_sizes)
        if any([t == 0 for t in pop_sizes]):
            raise Exception("All islands need to be populated")
        super().__init__(num_agents, MASK=MASK)
        self.pop_sizes = pop_sizes
        self.num_islands = len(pop_sizes)
        self.cum_islands = torch.cumsum(torch.tensor([0] + list(pop_sizes)), dim=0)
        self.distribution = torch.zeros(self.num_islands)
        # this will be a softmax, and will be updated though gradient backprop
        self.buffer = torch.zeros(self.num_islands)
        self.softmax_update = softmax_update

    def get_total_distribution(self,
                               T,
                               N=1,
                               init_team=None,
                               tracked=None,
                               **kwargs,
                               ):
        single_dist = self.single_member_distribution()
        dist = dict()
        for team in itertools.product(range(self.num_agents), repeat=T*N):
            prob = torch.prod(single_dist[team,])
            dist[team] = prob.item()
        return dist

    def single_member_distribution(self):

        # (num_agents) multinomial distribution for each index to update
        dist = torch.zeros(self.num_agents)
        for island_idx, prob in enumerate(torch.softmax(self.distribution, dim=-1).view(-1, 1)):
            dist[self.cum_islands[island_idx]:self.cum_islands[island_idx + 1]] = prob/self.pop_sizes[island_idx]
        return dist

    def get_member_distribution(self,
                                init_team,
                                indices=None,
                                **kwargs
                                ):
        if indices is None:
            indices = torch.where(torch.eq(init_team, self.MASK))
        if len(indices[0]) == 0:
            return None, None

        # (|indices|, num_agents) multinomial distribution for each index to update
        dist = self.single_member_distribution().unsqueeze(0).broadcast_to((len(indices[0]), self.num_agents))

        return indices, dist

    def global_indices_to_islands(self, team):
        """
        returns which island each index in an array is on
        Args:
            team: vector of population indices
        Returns:
            vector of island indices
        """
        team = team.unsqueeze(0)
        stuff = self.cum_islands[1:]
        while len(stuff.shape) < len(team.shape):
            stuff = stuff.unsqueeze(-1)

        return torch.sum(team >= stuff, dim=0)

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dic = {'buffer': self.buffer,
               'dist': self.distribution,
               }
        torch.save(dic, os.path.join(save_dir, 'stuff.pkl'))

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        dic = torch.load(os.path.join(save_dir, 'stuff.pkl'))
        self.buffer = dic['buffer']
        self.distribution = dic['dist']

    def train(self, *args, **kwargs):
        print('trainin')
        print(torch.softmax(self.distribution, dim=-1))
        if torch.sum(self.buffer) == 0:
            print('skippin')
            return
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
        print(torch.softmax(self.distribution, dim=-1))
        # reset buffer
        self.buffer = torch.zeros(self.num_islands)

    def add_to_buffer(self, scalar, obs_preembed, team, obs_mask, weight=1.):
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
        island_fill = torch.multinomial(torch.softmax(self.distribution, dim=-1),
                                        num_samples=len(indices[0]),
                                        replacement=True,
                                        )
        fill = torch.zeros_like(island_fill)
        for i in range(self.num_islands):
            island_indices = torch.where(torch.eq(island_fill, i))
            fill[island_indices] = torch.randint(self.cum_islands[i].item(),
                                                 self.cum_islands[i + 1].item(),
                                                 (len(island_indices[0]),)
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

        return self.fill_in_teams(initial_teams=self.create_masked_teams(T=T, N=N),
                                  obs_preembed=obs_preembed,
                                  obs_mask=obs_mask,
                                  noise_model=noise_model,
                                  )


if __name__ == '__main__':
    save_dir = 'temp/test'
    thing = MCAAMainland(pop_sizes=(2, 3, 3, 4), softmax_update=1)
    goal_team = torch.tensor([2, 3, 4, 5, 6, 7])
    goal_dist = thing.global_indices_to_islands(goal_team)
    print(goal_dist)
    print(torch.softmax(thing.distribution, dim=-1))
    for i in range(5000):
        thing.add_to_buffer(1, None, goal_team, None)
        thing.train()

    thing.add_to_buffer(1, None, goal_team, None)
    print('save test')
    print(thing.distribution)
    print(thing.buffer)
    thing.save(save_dir)
    thing.load(save_dir)
    print(thing.distribution)
    print(thing.buffer)
    print()
    print(torch.softmax(thing.distribution, dim=-1))
    print(thing.distribution)
    team_samp = thing.create_teams(10, N=4)
    island_samp = thing.global_indices_to_islands(team_samp)
    print(team_samp)
    print(island_samp)
    print([torch.sum(torch.eq(island_samp, i)).item()/len(island_samp.flatten()) for i in range(thing.num_islands)])

    fake_team = thing.create_masked_teams(10)
    fake_team[0, :5] = 69
    print(thing.fill_in_teams(initial_teams=fake_team))
