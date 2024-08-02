import torch
from src.team_trainer import DiscreteInputTrainer
from src.coevolver import CaptianCoevolution
from src.language_replay_buffer import ReplayBufferDiskStorage
from matplotlib import pyplot as plt
from coevolution_tests.rps_basic.game import plot_dist_evolution
from coevolution_tests.rps_basic.game import SingleOutcome


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

    DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))
    plot_dir = os.path.join(DIR, 'data', 'plots', 'tests_rps_teams')
    if not os.path.exists((plot_dir)):
        os.makedirs(plot_dir)
    trainer = DiscreteInputTrainer(
        buffer=ReplayBufferDiskStorage(storage_dir=os.path.join(DIR, 'data', 'test_storage'),
                                       capacity=1e5,
                                       device=None,
                                       ),
        num_agents=3,
        num_input_tokens=3,
        embedding_dim=64,
        pos_encode_input=False,
        num_decoder_layers=4,
        num_encoder_layers=4,
    )

    coevolver = CaptianCoevolution(population_sizes=[3],
                                   outcome_fn_gen=lambda: SingleOutcome(agents=torch.arange(3)),
                                   team_trainer=trainer,
                                   clone_fn=lambda: None,
                                   mutation_fn=None,
                                   captian_elo_update=.05
                                   )

    minibatch = 64
    init_dists = []
    cond_dists = []
    losses = []
    for epoch in range(100):
        noise = trainer.create_nose_model_towards_uniform(.1)
        coevolver.set_noise_model(noise_model=noise)
        coevolver.epoch(rechoose=False)

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
        loss = trainer.training_step(batch_size=minibatch, sgd=False)
        losses.append(loss)
        print('epoch', epoch, ';\tbuffer size', len(trainer.buffer))
        print('loss', loss)
        if not (epoch + 1)%10:
            loss_plot(losses=losses, save_dir=os.path.join(plot_dir, 'loss_plot.png'))
            plot_dist_evolution(init_dists, save_dir=os.path.join(plot_dir, 'init_dist.png'))
            for k, name in enumerate('RPS'):
                plot_dist_evolution([dist[k] for dist in cond_dists],
                                    save_dir=os.path.join(plot_dir, 'dist_against' + name + '.png'),
                                    title='Distribution against ' + name)
    trainer.buffer.clear()
