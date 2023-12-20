from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.algorithm import Algorithm

import environments.environment_creator

from ray.tune import Tuner

from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.ppo import PPO, PPOTorchPolicy

if __name__ == '__main__':
    data_frame_path: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_datasets/' + 'data_1'

    # path_to_checkpoint: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_results/PPO_2023-12-01_11-48-20/PPO_ALE_Pong-v5_24b2d_00000_0_2023-12-01_11-48-20/checkpoint_000930'
    storage_directory: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_results/PPO_2023-12-15_14-02-14'
    tuner: Tuner = Tuner.restore(path=storage_directory, trainable=PPO)
    result_grid = tuner.get_results()
    best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
    path_checkpoint: str = best_result.best_checkpoints[0][0].path

    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)

    # algorithm_config: AlgorithmConfig = algorithm.config.copy(copy_frozen=False)
    # # algorithm_config.environments(env='my_pong')
    # algorithm_config.rollouts(num_rollout_workers=0)
    # algorithm_config.evaluation(evaluation_num_workers=4)
    # algorithm: Algorithm = algorithm_config.build()
    # algorithm.restore(path_checkpoint)
    #
    # policy_id = algorithm.get_policy()._Policy__policy_id
    policy: PPOTorchPolicy = algorithm.get_policy()
    policy.compute_actions()

    selected_eval_worker_ids = algorithm.evaluation_workers.healthy_worker_ids()
    all_batches = []

    for i in range(100):
        batches = algorithm.evaluation_workers.foreach_worker(
            func=lambda w: w.sample(),
            local_worker=False,
            remote_worker_ids=selected_eval_worker_ids,
            timeout_seconds=algorithm.config.evaluation_sample_timeout_s,
        )
        total_batch = concat_samples(batches)
        batch = total_batch[policy_id]

    pass