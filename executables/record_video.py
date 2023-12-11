from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
import environments.environment_creator
from ray import tune
from ray.tune import Tuner

from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.ppo import PPO

if __name__ == '__main__':
    # storage_directory: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_results/DQN_2023-12-04_17-31-21'
    storage_directory: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/debug/ray_debug/DQN_2023-12-07_12-29-11'
    tuner: Tuner = Tuner.restore(path=storage_directory, trainable=DQN)
    result_grid = tuner.get_results()
    best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
    path_checkpoint: str = best_result.best_checkpoints[0][0].path

    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)

    algorithm_config: AlgorithmConfig = Algorithm.get_config(algorithm).copy(copy_frozen=False)
    algorithm_config.environment(env='record_video_pong')
    algorithm_config.rollouts(num_rollout_workers=0)
    algorithm_config.evaluation(
        evaluation_duration=1,
        evaluation_num_workers=2,
    )

    algorithm: Algorithm = algorithm_config.build()
    algorithm.restore(path_checkpoint)

    number_iteration: int = 10
    for i in range(number_iteration):
        algorithm.evaluate()
