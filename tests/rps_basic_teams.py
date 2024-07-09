import torch
from src.team_trainer import DiscreteInputTrainer
from src.language_replay_buffer import ReplayBufferDiskStorage
from matplotlib import pyplot as plt
from tests.rps_basic_game import outcomes, plot_dist_evolution

def loss_plot(losses, save_dir=None, show=False):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.title('CrossEntropy Loss')
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    import os, sys

    torch.random.manual_seed(69)

    # print(outcomes((torch.tensor([[0], [1], [2], [0]]), torch.tensor(([[1], [0], [0], [0]]))))[-1])

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
    plot_dir = os.path.join(DIR, 'data', 'plots', 'tests_rps')
    if not os.path.exists((plot_dir)):
        os.makedirs(plot_dir)
    trainer = DiscreteInputTrainer(num_agents=3,
                                   num_input_tokens=3,
                                   embedding_dim=64,
                                   pos_encode_input=False,
                                   num_decoder_layers=4,
                                   num_encoder_layers=4,
                                   )
    N = 100
    capacity = int(1e5)
    buffer = ReplayBufferDiskStorage(storage_dir=os.path.join(DIR, "data", "temp", "tests_rps"), capacity=capacity)

    minibatch = 64
    init_dists = []
    cond_dists = []
    losses = []
    for epoch in range(100):
        noise = trainer.create_nose_model_towards_uniform(.1)
        for scalar, preembed, team, mask in trainer.collect_training_data(outcome=outcomes,
                                                                          num_members=(1, 1),
                                                                          N=N,
                                                                          number_of_tie_matches=0,
                                                                          number_of_loss_rematches=1,
                                                                          noise_model=noise
                                                                          ):
            buffer.push((scalar, preembed, team, mask))

            # buffer.push((scalar, None, team, None))
        init_distribution = trainer.team_builder.forward(obs_preembed=None,
                                                         target_team=trainer.create_masked_teams(T=1, N=1),
                                                         obs_mask=None,
                                                         ).detach().flatten().numpy()
        init_dists.append(init_distribution)
        conditional_dists = []
        for opponent in range(3):
            dist = trainer.team_builder.forward(obs_preembed=torch.tensor([[opponent]]),
                                                target_team=trainer.create_masked_teams(T=1, N=1),
                                                obs_mask=None,
                                                )
            conditional_dists.append(dist.detach().flatten().numpy())
        cond_dists.append(conditional_dists)
        sample = buffer.sample(batch=minibatch)

        loss = trainer.training_step(data=sample, minibatch=False).item()
        losses.append(loss)
        print('epoch', epoch, ';\tbuffer size', len(buffer))
        print('loss', loss)
        if not (epoch + 1)%10:
            loss_plot(losses=losses, save_dir=os.path.join(plot_dir, 'loss_plot.png'))
            plot_dist_evolution(init_dists, save_dir=os.path.join(plot_dir, 'init_dist.png'))
            for k, name in enumerate('RPS'):
                plot_dist_evolution([dist[k] for dist in cond_dists],
                                    save_dir=os.path.join(plot_dir, 'dist_against' + name + '.png'),
                                    title='Distribution against ' + name)
    buffer.close()
