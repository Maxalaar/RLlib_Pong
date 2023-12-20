import ray
import torch
from ray import air, tune
from ray.rllib.algorithms import AlgorithmConfig

from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.utils.replay_buffers import MultiAgentPrioritizedReplayBuffer

if __name__ == '__main__':
    ray.init(num_gpus=1, local_mode=False)

    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available? {cuda_available}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {gpu_count}")

        current_gpu = torch.cuda.current_device()
        print(f"Current GPU: {current_gpu}")

        gpu_name = torch.cuda.get_device_name(current_gpu)
        print(f"GPU Name: {gpu_name}")

    algorithm_configuration: DQNConfig = (
        DQNConfig()
        .environment(env='ALE/Pong-v5', env_config={'frameskip': 1, 'full_action_space': False, 'repeat_action_probability': 0.0})
        .resources(num_gpus=0.5)
        .framework('torch')
        .rollouts(num_rollout_workers=4, num_envs_per_worker=5)
        .training(model={'grayscale': True, 'zero_mean': False, 'dim': 42}, replay_buffer_config={'type': MultiAgentPrioritizedReplayBuffer, 'prioritized_replay_alpha': 0.5, 'capacity': 50000})
        .exploration(explore=True, exploration_config={'initial_epsilon': 0.5, 'epsilon_timesteps': 1, 'final_epsilon': 0.1})
    )
    # algorithm_configuration.num_atoms = 51
    # algorithm_configuration.noisy = True
    # algorithm_configuration.gamma = 0.99
    # algorithm_configuration.lr = 0.0001
    # algorithm_configuration.hiddens = [512]
    # algorithm_configuration.rollout_fragment_length = 4
    # algorithm_configuration.train_batch_size = 32
    # algorithm_configuration.target_network_update_freq = 500
    # algorithm_configuration.num_steps_sampled_before_learning_starts = 10_000
    # algorithm_configuration.n_step = 3
    # algorithm_configuration.compress_observations = True

    tuner = tune.Tuner(
        trainable='DQN',
        param_space=algorithm_configuration,
        run_config=air.RunConfig(
            storage_path='/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/debug/ray_results',
            stop={
                'episode_reward_mean': 1000,
            },
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_score_order='max',
                checkpoint_frequency=50,
            )
        ),
    )

    tuner.fit()
