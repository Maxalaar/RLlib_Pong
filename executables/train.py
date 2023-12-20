import ray
import torch
from ray import air, tune
from ray.rllib.algorithms import AlgorithmConfig

from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig

# import environments.environment_creator
# import models.register_model

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

    algorithm_configuration: AlgorithmConfig = (
        # PPOConfig()
        DQNConfig()
        # .environment('my_pong')
        # .environment('CartPole-v1')
        .environment('ALE/Pong-v5', env_config={'frameskip': 1, 'full_action_space': False, 'repeat_action_probability': 0.0})
        .rollouts(rollout_fragment_length='auto', batch_mode='truncate_episodes', observation_filter='NoFilter')    # num_rollout_workers=4, num_envs_per_worker=5, 
        .resources(num_gpus=0.5)      # num_gpus_per_worker=0.1, num_gpus_per_learner_worker=0.1,
        .framework('torch')
        # .training(model={'fcnet_hiddens': [256, 256]})     # 'fcnet_hiddens': [256, 256], "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]],  model={'custom_model': 'custom_cnn'}
        # .evaluation(evaluation_num_workers=1, evaluation_interval=50)
    )
    if type(algorithm_configuration) is DQNConfig:
        algorithm_configuration: DQNConfig
        trainable: str = 'DQN'
        # algorithm_configuration.training_intensity = None
        # For Rainbow
        algorithm_configuration.n_step = 5
        algorithm_configuration.noisy = True
        algorithm_configuration.num_atoms = 51
        algorithm_configuration.v_min = -21
        algorithm_configuration.v_max = 21
    elif type(algorithm_configuration) is PPOConfig:
        algorithm_configuration: PPOConfig
        trainable: str = 'PPO'
    else:
        raise ValueError('Algorithm is not supported')

    tuner = tune.Tuner(
        trainable=trainable,
        param_space=algorithm_configuration,
        run_config=air.RunConfig(
            # storage_path='/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/debug/ray_results',
            storage_path='/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_results/',
            stop={
                # 'time_total_s': 60 * 60 * 24,
                'episode_reward_mean': 1000,
            },
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_score_order='max',
                checkpoint_frequency=60,
            )
        ),
    )

    tuner.fit()
