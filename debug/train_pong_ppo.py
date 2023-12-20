import ray
import torch
from ray import air, tune
from ray.rllib.algorithms import AlgorithmConfig

from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig

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
        PPOConfig()
        .environment(env='ALE/Pong-v5', env_config={'frameskip': 1, 'full_action_space': False, 'repeat_action_probability': 0.0})
        # .environment('CartPole-v1')
        .rollouts(num_rollout_workers=4, num_envs_per_worker=5, rollout_fragment_length='auto', batch_mode='truncate_episodes', observation_filter='NoFilter')
        .resources(num_gpus=0.5)
        .framework('torch')
        .training(lambda_=0.95, kl_coeff=0.5, clip_param=0.1, vf_clip_param=10.0, entropy_coeff=0.01, train_batch_size=5000, sgd_minibatch_size=500, num_sgd_iter=10, model={'dim': 42, 'vf_share_layers': True})
    )

    tuner = tune.Tuner(
        trainable='PPO',
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
                checkpoint_frequency=60,
            )
        ),
    )

    tuner.fit()
