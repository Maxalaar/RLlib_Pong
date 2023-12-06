from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
import environment.environment_creator

if __name__ == '__main__':
    path_checkpoint: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_results/DQN_2023-12-04_17-31-21/DQN_my_pong_8ee3c_00000_0_2023-12-04_17-31-21/checkpoint_002829'
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
