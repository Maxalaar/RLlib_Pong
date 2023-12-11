from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.policy.sample_batch import concat_samples

import pandas as pd
import h5py

import environments.environment_creator

from ray.tune import Tuner

from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.ppo import PPO


def save_data_in_data_frame(path, new_observations, new_actions):
    with h5py.File(path + '.h5', 'a') as hf:
        if 'observations' not in hf:
            # Create a dataset for observations if it doesn't exist
            hf.create_dataset('observations', data=new_observations, chunks=True, maxshape=(None, *new_observations.shape[1:]))
            hf.create_dataset('actions', data=new_actions, chunks=True, maxshape=(None,))
        else:
            # If the dataset exists, append new data
            observations_dset = hf['observations']
            actions_dset = hf['actions']

            # Resize the dataset to accommodate new data
            observations_dset.resize((observations_dset.shape[0] + new_observations.shape[0]), axis=0)
            actions_dset.resize((actions_dset.shape[0] + new_actions.shape[0]), axis=0)

            # Append the new data
            observations_dset[-new_observations.shape[0]:] = new_observations
            actions_dset[-new_actions.shape[0]:] = new_actions


if __name__ == '__main__':
    data_frame_path: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_datasets/' + 'data_1'

    # path_to_checkpoint: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_results/PPO_2023-12-01_11-48-20/PPO_ALE_Pong-v5_24b2d_00000_0_2023-12-01_11-48-20/checkpoint_000930'
    storage_directory: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/debug/ray_debug/DQN_2023-12-07_12-29-11'
    tuner: Tuner = Tuner.restore(path=storage_directory, trainable=DQN)
    result_grid = tuner.get_results()
    best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')
    path_checkpoint: str = best_result.best_checkpoints[0][0].path

    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)

    algorithm_config: AlgorithmConfig = algorithm.config.copy(copy_frozen=False)
    # algorithm_config.environments(env='my_pong')
    algorithm_config.rollouts(num_rollout_workers=0)
    algorithm_config.evaluation(evaluation_num_workers=4)
    algorithm: Algorithm = algorithm_config.build()
    algorithm.restore(path_checkpoint)

    policy_id = algorithm.get_policy()._Policy__policy_id
    selected_eval_worker_ids = algorithm.evaluation_workers.healthy_worker_ids()
    all_batches = []

    for i in range(10):
        batches = algorithm.evaluation_workers.foreach_worker(
            func=lambda w: w.sample(),
            local_worker=False,
            remote_worker_ids=selected_eval_worker_ids,
            timeout_seconds=algorithm.config.evaluation_sample_timeout_s,
        )
        total_batch = concat_samples(batches)
        batch = total_batch[policy_id]
        save_data_in_data_frame(path=data_frame_path, new_observations=batch['obs'], new_actions=batch['actions'])

        print(i)
        print('total_batch.count : ' + str(total_batch.count))
        print('')


        # all_batches.extend(batches)
        # total_batch = concat_samples(all_batches)
        # all_batches.clear()
        #
        # print(i)
        # print('total_batch.count : ' + str(total_batch.count))
        # print('')
        # writer.write(total_batch[policy_id])
        # writer.write(total_batch[policy_id])