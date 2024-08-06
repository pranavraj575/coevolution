from unstable_baselines3.utils.dict_keys import *

# whether agent is able to be trained/should be trained
# DICT_TRAIN = DICT_TRAIN


# keeps track of age of each agent
DICT_AGE = 'age'

# keeps track of time since last mutation for each agent
DICT_MUTATION_AGE = 'mutation_age'

# keeps track of whether each agent is a worker
# i.e. inherits multi_agent_algs/off_policy:OffPolicy or multi_agent_algs/on_policy:OnPolicy
DICT_IS_WORKER = 'is_worker'

# whether an agent should only collect buffer examples, usually implies DICT_TRAIN is false
# If using this, should probably have the following setup:
#   all DICT_TRAIN for a particular agent is false
#   all DICT_COLLECT_ONLY for that agent is true
# this will result in the agent only training in one large batch after each epoch
# this should be done automatically if  coevolver:PettingZooCaptianCoevolution.local_collection_mode
DICT_COLLECT_ONLY = 'collect_only'

# whether agent is clonable
DICT_CLONABLE = 'clonable'

# whether agent can be replaced by a clone
DICT_CLONE_REPLACABLE = 'clone_replacable'

# whether agent can be replaced by a mutation
DICT_MUTATION_REPLACABLE = 'mutation_replacable'

# when replacing, whether agent should update with the old agent's buffe
# i.e. the info dict of the agent that is about to replace the other agent determines this
DICT_UPDATE_WITH_OLD_BUFFER = 'update_with_old_buffer'

# when replacing, whether agent should keep the old agent's buffer (overrides previous)
DICT_KEEP_OLD_BUFFER = 'keep_old_buffer'

# keys that are position dependent
# i.e. if agent is replaced with a clone or mutated, these keys will be unchanged
# by default, this is {DICT_CLONE_REPLACABLE, DICT_MUTATION_REPLACABLE}
DICT_POSITION_DEPENDENT = 'position_dependent'

# whether to save class of agent (mostly useless, and should always be true)
DICT_SAVE_CLASS = 'save_class'

# whether to save buffer of agent (mostly useless, and should always be true)
DICT_SAVE_BUFFER = 'save_buffer'

###### these keys are reassigned upon each trial

# whether an agent is a captian
TEMP_DICT_CAPTIAN = 'captian'

# whether a captain is unique
TEMP_DICT_CAPTIAN_UNIQUE = 'unique'

# team id, member id
TEMP_DICT_TEAM_MEMBER_ID = 'team_member_id'

###### these keys are for coevolution dict

COEVOLUTION_DICT_ELOS = 'elos'
COEVOLUTION_DICT_CAPTIAN_ELO_UPDATE = 'captian_elo_update'
COEVOLUTION_DICT_MEMBER_ELO_UPDATE = 'member_elo_update'
COEVOLUTION_DICT_ELO_CONVERSION = 'elo_conversion'
COEVOLUTION_DICT_DEPTH_OF_RETRY = 'depth_of_retry'

__all__ = ["DICT_TRAIN",
           "DICT_AGE",

           "DICT_IS_WORKER",

           "DICT_CLONABLE",
           "DICT_CLONE_REPLACABLE",
           "DICT_MUTATION_REPLACABLE",
           "DICT_COLLECT_ONLY",

           "DICT_SAVE_BUFFER",
           "DICT_SAVE_CLASS",

           "TEMP_DICT_CAPTIAN",
           "TEMP_DICT_CAPTIAN_UNIQUE",

           "DICT_POSITION_DEPENDENT",
           "DICT_KEEP_OLD_BUFFER",
           "DICT_UPDATE_WITH_OLD_BUFFER",

           "COEVOLUTION_DICT_ELOS",
           "COEVOLUTION_DICT_CAPTIAN_ELO_UPDATE",
           "COEVOLUTION_DICT_ELO_CONVERSION",
           ]
