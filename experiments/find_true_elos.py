import os, sys, pickle, torch
import numpy as np

from experiments.pyquaticus_utils.dist_plot import deorder_total_dist

DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
save_dir = os.path.join(DIR, 'data', 'save')

# output of pyquaticus_n_member_fixed_tournament.py
elo_ref_file = os.path.join(save_dir,
                            'basic_team_tournament',
                            'elos_torunament_team_size_2_arena_200_0__100_0.pkl',
                            )
# output of pyquaticus_teams_coevolution.py
ident = 'pyquaticus_coev_MLM_agents__arch_64_64_ppo_50__protect_1000_mut_prob_0_0_clone_1_elite_3_tm_sz_2_arena_200_0__100_0_kpout_15_0_embed_dim_128_trans__head_4_enc_3_dec_3_drop_0_1_inp_emb__lyrs_2_train_frq_5_btch_512_minibtch_64_half_life_400_0'
trial_file = os.path.join(save_dir,
                          ident,
                          'elo_trials.pkl',
                          )

total_dist_file = os.path.join(save_dir,
                               ident,
                               'total_dist.pkl',
                               )
output_file = os.path.join(save_dir,
                           ident,
                           'elos_of_all_teams.pkl')
plot_file = os.path.join(save_dir,
                         ident,
                         'elo_against_occurrence.png')

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
elo_dict = dict()

# just do grad descent on whole dataset
teams = sorted(elo_trials.keys())
team_to_idx = {team: i for i, team in enumerate(teams)}
if os.path.exists(output_file):
    dic = torch.load(output_file)
    elos = torch.nn.Parameter(dic['elos'])
    optim = torch.optim.Adam(params=torch.nn.ParameterList([elos]))
    optim.load_state_dict(dic['optim'])

    ###plotting
    if os.path.exists(total_dist_file):
        print('plotting')
        import matplotlib.pyplot as plt

        f = open(total_dist_file, 'rb')
        total_dist = pickle.load(f)
        f.close()

        total_dist = deorder_total_dist(total_dist)
        plt.scatter([elos[team_to_idx[team]].item()*elo_conversion + 1000 for team in teams],
                    [total_dist[team] for team in teams],
                    )
        plt.xlabel('True Elo')
        plt.ylabel('BERTeam occurrence probability')
        plt.savefig(plot_file)
        plt.close()
        print('done plotting')
    else:
        print('total dist not available, run pyquaticus_teams_coevolution')


else:
    elos = torch.nn.Parameter(torch.zeros(len(elo_trials)))
    optim = torch.optim.Adam(params=torch.nn.ParameterList([elos]))
old_loss = torch.nan
for epoch in range(1000):
    optim.zero_grad()
    loss = torch.zeros(1)
    for team in elo_trials:
        this_loss = torch.zeros(1)
        count = 0
        for trial in elo_trials[team]:
            for opp in trial:
                opp_elo = elo_ref[opp]
                expectation = 1/(1 + torch.exp(opp_elo - elos[team_to_idx[team]]))
                actual = torch.tensor(trial[opp]['team result'], dtype=torch.float)
                this_loss += torch.nn.MSELoss().forward(expectation, actual)
                count += 1
        this_loss = this_loss/count
        loss += this_loss
    loss.backward()
    optim.step()
    # print('inc loss diff:', (loss - old_loss).item()/len(teams))
    print('epoch:', epoch, ';', 'avg loss:', loss.item()/len(teams),
          end='                                \r\r')
    old_loss = loss.item()
    if not (epoch + 1)%10:
        print('saving',
              end='                                \r')
        dic = {
            'elos': elos.detach(),
            'optim': optim.state_dict(),
        }
        torch.save(dic, output_file)
        print('done saving',
              end='                                \r')

quit()

for team in elo_trials:
    print(team)
    elo = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.Adam(params=torch.nn.ParameterList([elo]))
    while True:
        optim.zero_grad()
        loss = torch.zeros(1)
        count = 0
        for trial in elo_trials[team]:
            for opp in trial:
                opp_elo = elo_ref[opp]
                expectation = 1/(1 + torch.exp(opp_elo - elo))
                actual = torch.tensor(trial[opp]['team result'], dtype=torch.float)
                this_loss = torch.nn.MSELoss().forward(expectation, actual)
                loss += this_loss
                count += 1
                # print('result of', trial[opp]['team result'],
                #      'against', opp, '(elo', elo_ref[opp]*elo_conversion + 1000, ')')
        loss = loss/count
        loss.backward()
        optim.step()
        print(loss.item())
