import os, sys, pickle, torch, time
import numpy as np

from experiments.pyquaticus_utils.dist_plot import deorder_total_dist, order_compensate_dist

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
aggression_file = os.path.join(save_dir,
                               ident,
                               'aggression.pkl',
                               )

output_file = os.path.join(save_dir,
                           ident,
                           'elos_of_all_teams.pkl')
plot_dir = os.path.join(save_dir,
                        ident,
                        'plots', )
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
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

f = open(total_dist_file, 'rb')
total_dist = pickle.load(f)
f.close()

elo_dict = dict()

# just do grad descent on whole dataset
teams = sorted(elo_trials.keys())
team_size = 2
team_to_idx = {team: i for i, team in enumerate(teams)}
if os.path.exists(output_file):
    dic = torch.load(output_file)
    elos = torch.nn.Parameter(dic['elos'])
    optim = torch.optim.Adam(params=torch.nn.ParameterList([elos]),
                             # lr=.01,
                             )
    optim.load_state_dict(dic['optim'])

    ###plotting
    if os.path.exists(total_dist_file):
        print('plotting')
        import matplotlib.pyplot as plt

        plt.rcParams.update({'font.size': 14})

        f = open(aggression_file, 'rb')
        aggression = pickle.load(f)
        f.close()
        aggression = [agg > 1 for agg in aggression]
        for compensate in False, True:
            if compensate:
                s = 'compensated_'
                total_deordered_dist = deorder_total_dist(order_compensate_dist(total_dist))
            else:
                s = ''
                total_deordered_dist = deorder_total_dist(total_dist)

            split_agg_agents = [[[], []] for _ in range(team_size + 1)]
            # labels = [str(agg_agents) + ' aggressive' for agg_agents in range(len(split_agg_agents))]
            labels = ['Defensive', 'Balanced', 'Aggressive']
            all_agg_agents = [[], []]
            for team in teams:
                conv_elo = elos[team_to_idx[team]].item()*elo_conversion + 1000
                occurrence = total_deordered_dist[team]
                # number of aggressive agents
                index = sum([aggression[idx] for idx in team])
                split_agg_agents[index][0].append(conv_elo)
                split_agg_agents[index][1].append(occurrence)
                all_agg_agents[0].append(conv_elo)
                all_agg_agents[1].append(occurrence)

            x = np.array(all_agg_agents[0])
            y = np.array(all_agg_agents[1])
            model = np.polyfit(x,
                               y,
                               1,  # linear
                               )
            predict = np.poly1d(model)

            ssresid = np.sum(np.square(y - predict(x)))
            sstotal = np.sum(np.square(y - np.mean(y)))
            r2 = 1 - ssresid/sstotal
            print('linear fit')
            print('y = ' + str(model[0]) + ' x + ' + str(model[1]))
            print('r2 error', r2)

            for agg_agents, (x_part, y_part) in enumerate(split_agg_agents):
                aggression_amount = agg_agents/team_size
                red = int(aggression_amount*255)
                blue = 255 - red
                red = ('0' + hex(red)[2:])[-2:]
                blue = ('0' + hex(blue)[2:])[-2:]
                color = '#' + red + '00' + blue
                plt.scatter(x_part, y_part,
                            label=labels[agg_agents],
                            color=color,
                            s=10,
                            )

            xlim = plt.xlim()
            ylim = (0, plt.ylim()[1])
            # too much info in one plot
            plt.plot(xlim, predict(np.array(xlim)), label='linear fit', color='green', )

            plt.legend()
            plt.title('BERTeam Total Distribution vs True Elo')
            plt.xlabel('True Elo')
            plt.ylabel('Occurrence Probability')
            plt.ylim(ylim)
            plt.xlim(xlim)
            plt.savefig(os.path.join(plot_dir,
                                     s + 'split_elo_against_occurrence.png'),
                        bbox_inches='tight',
                        )
            plt.close()

            plt.scatter(x, y,
                        s=10,
                        )
            xlim = plt.xlim()
            ylim = (0, plt.ylim()[1])


            def form(s):
                mult, exp = ('{:0.2e}'.format(s)).split('e')
                return mult + '\\cdot' + '10^{' + exp + '}'


            plt.plot(xlim, predict(np.array(xlim)),
                     label=('linear fit; $R^2=' + str(round(r2, 2)) + '$'),
                     # label=('$y=' + form(model[0]) + 'x' + form(model[1]) +
                     #       '$; $R^2=' + str(round(r2, 2)) + '$'),
                     color='purple',
                     )
            plt.title('BERTeam Total Distribution vs True Elo')
            plt.xlabel('Elo')
            plt.ylabel('Occurrence Probability')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.legend()
            plt.savefig(os.path.join(plot_dir,
                                     s + 'elo_against_occurrence.png'),
                        bbox_inches='tight',
                        )
            plt.close()
        print('done plotting')
    else:
        print('total dist not available, run pyquaticus_teams_coevolution')
else:
    elos = torch.nn.Parameter(torch.zeros(len(elo_trials)))
    optim = torch.optim.Adam(params=torch.nn.ParameterList([elos]))

manual = True  # use standard elo update instead of torch gradients (faster)
if manual:
    elos = elos.detach()

old_loss = torch.nan
for epoch in range(1000):
    start = time.time()
    if not manual:
        optim.zero_grad()
    loss = torch.zeros(1)
    old_elos = elos.detach().clone()
    for team in elo_trials:
        this_loss = torch.zeros(1)
        count = 0
        for trial in elo_trials[team][:2]:
            # for trial in elo_trials[team][:2]+elo_trials[team][-2:]:
            for opp in trial:
                opp_elo = elo_ref[opp]
                expectation = 1/(1 + torch.exp(opp_elo - elos[team_to_idx[team]]))
                actual = torch.tensor(trial[opp]['team result'], dtype=torch.float)
                if manual:
                    elos[team_to_idx[team]] += .1*(actual - expectation)
                if not manual:
                    this_loss += torch.nn.MSELoss().forward(expectation, actual)
                count += 1
        this_loss = this_loss/count
        loss += this_loss
    if not manual:
        loss.backward()
        optim.step()
    # print('inc loss diff:', (loss - old_loss).item()/len(teams))
    mean_update_mag = torch.mean(torch.abs(old_elos - elos)).item()
    print('epoch:', epoch, ';',
          '' if manual else ' '.join(['avg loss:', str(round(loss.item()/len(teams), 3)), ';', ]),
          '' if manual else ' '.join(['inc loss:', str((loss - old_loss).item()/len(teams)), ';', ]),
          'update mag:', mean_update_mag, ';',
          'time:', round(time.time() - start),
          end='                                \r\r')
    old_loss = loss.item()
    if epoch == 0 and mean_update_mag == 0:
        print('\nfinished, update magnitude is 0')
        break
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
        if mean_update_mag == 0:
            print('\nfinished, update magnitude is 0')
            break
for compensate in False, True:
    print('compensating:', compensate)
    print('average elo', torch.mean(elos).item()*elo_conversion + 1000)
    if compensate:
        temp_dist = order_compensate_dist(total_dist)
    else:
        temp_dist = total_dist
    expected_elo = 0
    for team in teams:
        elo = elos[team_to_idx[team]].item()
        prob = temp_dist[team]
        expected_elo += elo*prob
    print('expected elo', expected_elo*elo_conversion + 1000)
    print('max elo', torch.max(elos).item()*elo_conversion + 1000)
    argmax_team = max(teams, key=lambda team: temp_dist[team])
    print('mle', argmax_team)
    print('MLE elo', elos[team_to_idx[argmax_team]].item()*elo_conversion + 1000)

    bound = 950
    idxs_over = torch.where(elos*elo_conversion + 1000 > bound)[0]
    print('count over', bound, len(idxs_over))
    print('prob over', bound, sum([temp_dist[teams[idx]] for idx in idxs_over]))
