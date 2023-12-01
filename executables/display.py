import gymnasium
from gymnasium.wrappers import RecordVideo

import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.tune.registry import register_env


def pong_creator(environment_configuration):
    return gymnasium.make(id='ALE/Pong-v5', render_mode=environment_configuration['render_mode'])


def record_video_pong_creator(environment_configuration):
    environment = gymnasium.make(id='ALE/Pong-v5', render_mode='rgb_array')
    wrapped_env = RecordVideo(environment, video_folder='./ray_videos')
    return wrapped_env


if __name__ == '__main__':
    register_env(name='my_pong', env_creator=pong_creator)
    register_env(name='record_video_pong', env_creator=record_video_pong_creator)

    path_checkpoint: str = '/ray_results/PPO_2023-11-30_16-50-38/PPO_ALE_Pong-v5_35119_00000_0_2023-11-30_16-50-38/checkpoint_000194'
    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)
    algorithm_config: AlgorithmConfig = Algorithm.get_config(algorithm).copy(copy_frozen=False)

    algorithm_config.environment(env='record_video_pong', env_config={'render_mode': 'rgb_array'})
    algorithm_config.evaluation(
        evaluation_duration=1,
        evaluation_config={'render_mode': 'rgb_array'},
    )
    algorithm_config.rollouts(
        num_rollout_workers=1,
        num_envs_per_worker=1,
    )

    algorithm: Algorithm = algorithm_config.build()
    algorithm.restore(path_checkpoint)
    algorithm.evaluate()

    # number_iteration: int = 20
    # for i in range(number_iteration):
    #     evaluation_result = algorithm.evaluate()
