import ray
from ray.tune import Tuner
from ray.rllib.algorithms.dqn import DQN

import environment.environment_creator

if __name__ == '__main__':
    if ray.is_initialized():
        ray.shutdown()
    ray.init(local_mode=False)
    storage_directory: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_results/DQN_2023-12-04_17-31-21'
    tuner: Tuner = Tuner.restore(path=storage_directory, trainable=DQN)     # or PPO
    tuner.fit()
