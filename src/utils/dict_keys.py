from multi_agent_algs.utils.dict_keys import *

# keeps track of age of each agent
DICT_AGE = 'age'

# keeps track of whether each agent is a worker
# i.e. inherits multi_agent_algs/off_policy:OffPolicy or multi_agent_algs/on_policy:OnPolicy
DICT_IS_WORKER = 'is_worker'

# whether an agent should only collect buffer examples, usually implies DICT_TRAIN is false
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


###### these keys are for coevolution dict

COEVOLUTION_DICT_CAPTIAN_ELO='captian_elos'
COEVOLUTION_DICT_ELO_UPDATE='elo_update'
COEVOLUTION_DICT_ELO_CONVERSION='elo_conversion'
