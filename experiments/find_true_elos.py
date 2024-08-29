import os, sys, pickle
import numpy as np

DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
save_dir = os.path.join(DIR, 'data', 'save')

# output of pyquaticus_n_member_fixed_tournament.py
elo_ref_file = os.path.join(save_dir,
                            'basic_team_tournament',
                            'elos_torunament_team_size_2_arena_200_0__100_0.pkl',
                            )
# output of pyquaticus_teams_coevolution.py
trial_file = os.path.join(save_dir,
                          'pyquaticus_coev_MLM_agents__arch_64_64_ppo_50__protect_1000_mut_prob_0_0_clone_1_elite_3_tm_sz_2_arena_200_0__100_0_kpout_15_0_embed_dim_128_trans__head_4_enc_3_dec_3_drop_0_1_inp_emb__lyrs_2_train_frq_5_btch_512_minibtch_64_half_life_400_0',
                          'elo_trials.pkl',
                          )
elo_conversion = 400/np.log(10)
f = open(elo_ref_file, 'rb')
elo_ref = pickle.load(f)
# elo ref is a key (team -> elo) where team is a sorted tuple such as (2,5) and elo is a real number
# elos are unscaled and centered at 0
f.close()

f = open(trial_file, 'rb')
elo_trials = pickle.load(f)
# elo trials are a dict in the following format:
# (team indices, sorted): [{opponent indices, sorted: {
#                                                   'team result': team result,
#                                                   'opponent_ids': identity of opponents, if known
#                                                   } }]
#
f.close()
for team in elo_trials:
    print(team)
    for trial in elo_trials[team]:
        for opp in trial:
            print('result of', trial[opp]['team result'],
                  'against', opp, '(elo', elo_ref[opp]*elo_conversion + 1000, ')')
