import ray
import torch
from ray import air, tune

from ray.rllib.algorithms.ppo import PPOConfig

if __name__ == '__main__':
    ray.init(num_gpus=1)

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

    algorithm_configuration = (
        PPOConfig()
        .environment("ALE/Pong-v5")
        # .environment("CartPole-v1")
        .rollouts(num_rollout_workers=3)
        .resources(num_gpus_per_worker=0.1, num_gpus_per_learner_worker=0.1, num_gpus=0.5)    # num_gpus_per_worker=0,
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1)
    )

    tuner = tune.Tuner(
        trainable='PPO',
        param_space=algorithm_configuration,
        run_config=air.RunConfig(
            storage_path='/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_results',
            stop={
                # 'time_total_s': 60 * 60 * 24,
                'episode_reward_mean': 100,
            },
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_score_order='max',
                checkpoint_frequency=10,
            )
        ),
    )

    tuner.fit()