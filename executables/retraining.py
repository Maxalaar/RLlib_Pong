import ray
from ray.tune import Tuner
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.ppo import PPO

import environments.environment_creator

if __name__ == '__main__':
    if ray.is_initialized():
        ray.shutdown()
    ray.init(local_mode=False)
    storage_directory: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_results/DQN_2023-12-20_08-11-21'
    tuner: Tuner = Tuner.restore(path=storage_directory, trainable=DQN)     # or PPO
    tuner.fit()
