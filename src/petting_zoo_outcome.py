import torch, os

from src.zoo_cage import ZooCage
from src.utils.dict_keys import (DICT_IS_WORKER,
                                 DICT_TRAIN,
                                 DICT_COLLECT_ONLY,
                                 DICT_SAVE_BUFFER,
                                 DICT_SAVE_CLASS,
                                 TEMP_DICT_TEAM_MEMBER_ID,
                                 )
from BERTeam.outcome import PlayerInfo, OutcomeFn
from src.utils.savele_baselines import overwrite_worker


class PettingZooOutcomeFn(OutcomeFn):
    """
    outcome function which loads rl agents saved as files in specified directory
    """

    def __init__(self):
        super().__init__()
        self.local_mem = dict()
        self.counter = 0

    def set_zoo_dirs_and_pop_sizes(self, zoo_dirs, population_sizes):
        """
        Args:
            zoo_dirs: list of directories to find ZooCages
        """
        self.zoo = [ZooCage(zoo_dir=zoo_dir, overwrite_zoo=False) for zoo_dir in zoo_dirs]
        self.population_sizes = population_sizes

    def index_conversion(self, global_idx):
        pop_idx = 0
        while global_idx - self.population_sizes[pop_idx] >= 0:
            global_idx -= self.population_sizes[pop_idx]
            pop_idx += 1
        return pop_idx, global_idx

    def index_to_agent(self, idx, training_dict):
        """
        takes index and agent training dict and returns agent
            assumes self.set_zoo and self.set_index_conversion have already been called
        Args:
            idx: global index of agent
            training_dict: training dictionary associated with agent
                includes whether agent is captian, whether there are repeats, etc

        Returns:

        """
        pop_idx, local_idx = self.index_conversion(idx)
        collect_only = training_dict.get(DICT_COLLECT_ONLY, False)

        agent, saved_info = self.zoo[pop_idx].load_animal(key=str(local_idx),
                                                          load_buffer=not collect_only,
                                                          )
        return agent

    def get_outcome(self, team_choices, agent_choices, updated_train_infos=None, env=None):
        """
        Args:
            team_choices: K-tuple of teams, each team is an array of players
                players are indices corresponding
            agent_choices: same shape as team_choices, calculated agents (if applicable)
            updated_train_infos: either None or array of same shape as team_choices
                each element is dictionary of training settings for corresponding agent

                relevant keys:
                    DICT_TRAIN: whether the agent should be trained
                    DICT_COLLECT_ONLY: whether the agent should only collect data and update main agent's buffer
                    DICT_SAVE_BUFFER: whether to save buffer of agent (should probably be unspecified or true)
                    DICT_SAVE_CLASS: whether to save class of agent (should probably be unspecified or true)

        Returns: list corresponding to teams
            [
                outcome score,
                list corresponding to players of PlayerInfo(
                    obs_preembed=player observation (None or size (S,*) seq of observations);
                    obs_mask=observation mask (None or size (S,) boolean array of which items to mask;
                    )
            ]
        """

        if updated_train_infos is None:
            updated_train_infos = [[dict() for _ in team] for team in team_choices]
        if agent_choices is None:
            agent_choices = [
                [self.index_to_agent(member.item(), member_training) for member, member_training in zip(*t)]
                for t in zip(team_choices, updated_train_infos)]

        index_choices = [[member.item() for member in t] for t in team_choices]
        out = self._get_outcome_from_agents(agent_choices=agent_choices,
                                            index_choices=index_choices,
                                            updated_train_infos=updated_train_infos,
                                            env=env,
                                            )
        self._save_agents_to_local_mem(agent_choices=agent_choices,
                                       index_choices=index_choices,
                                       updated_train_infos=updated_train_infos,
                                       )
        return out

    def pop_local_mem(self):
        """
        returns local memory and clears it
        Returns: dict (global idx -> [(trained agent, updated agent info)]
            if the same agent appears multiple times, the respective list will have multiple elements
        """
        local_local_mem = self.local_mem
        self.local_mem = dict()
        self.counter = 0
        return local_local_mem

    def _save_agents_to_local_mem(self, agent_choices, index_choices, updated_train_infos):
        """
        saves trained agents to local memory for another class to pop and save
        Args:
            agent_choices:
            index_choices:
            updated_train_infos:
        Returns:
        """
        if self.dir is None:
            print('WARNING, DIRECTORY NOT SET, AGENTS CANNOT BE SAVED, AND POP_LOCAL_MEM WILL RETURN NOTHING')
            return
        for team_idx, t in enumerate(zip(agent_choices, index_choices, updated_train_infos)):
            for member_idx, (agent, global_idx, updated_train_dict) in enumerate(zip(*t)):
                is_worker = updated_train_dict.get(DICT_IS_WORKER, True)
                agent_dir = os.path.join(self.dir, str(self.ident), str(self.counter))
                if not os.path.exists(agent_dir):
                    os.makedirs(agent_dir)
                if is_worker:
                    overwrite_worker(worker=agent,
                                     worker_info=updated_train_dict,
                                     save_dir=agent_dir,
                                     save_buffer=updated_train_dict.get(DICT_SAVE_BUFFER, True),
                                     save_class=updated_train_dict.get(DICT_SAVE_CLASS, True),
                                     )
                    agent = None
                # else:
                #    agent = None
                if global_idx not in self.local_mem:
                    self.local_mem[global_idx] = []
                updated_train_dict[TEMP_DICT_TEAM_MEMBER_ID] = (team_idx, member_idx)
                self.local_mem[global_idx].append((agent_dir, agent, updated_train_dict))
                self.counter += 1

    def _get_outcome_from_agents(self, agent_choices, index_choices, updated_train_infos, env):
        """
        from workers, indices, and training info, evaluates the teams in a pettingzoo enviornment and returns
            the output for self.get_outcome
        also should train agents as specified by train_info
        also should save agents as specified by train_info
        Args:
            agent_choices: agents to use for training
            index_choices: global indices of agents
            updated_train_infos: info dicts, same shape as agetns
            env: env to use
        Returns:
            same output as self.get_outcome
            list corresponding to teams
            [
                outcome score,
                list corresponding to players of PlayerInfo(
                    obs_preembed=player observation (None or size (S,*) seq of observations);
                    obs_mask=observation mask (None or size (S,) boolean array of which items to mask;
                    )
            ]
        """
        raise NotImplementedError


if __name__ == '__main__':
    test0 = PlayerInfo()
    test = PlayerInfo(obs_preembed=torch.rand((1, 2, 3)))
    test2 = PlayerInfo(obs_preembed=torch.rand((2, 2, 3)))
    print(test0)
    print(test)
    print(test0.union_obs(test))
    print(test.union_obs(test))
    print(test.union_obs(test2))
    print(test0 == test)
    print(test0.union_obs(test) == test)
    print(test.union_obs(test0) == test)
