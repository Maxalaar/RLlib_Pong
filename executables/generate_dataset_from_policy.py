import ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.policy.sample_batch import concat_samples
from ray.rllib.offline.json_reader import JsonReader

import environment.environment_creator

if __name__ == '__main__':
    path_to_checkpoint: str = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_results/DQN_2023-12-04_17-31-21/DQN_my_pong_8ee3c_00000_0_2023-12-04_17-31-21/checkpoint_002847'
    writer = JsonWriter('./ray_dataset/')

    algorithm: Algorithm = Algorithm.from_checkpoint(path_to_checkpoint)
    algorithm_config: AlgorithmConfig = algorithm.config.copy(copy_frozen=False)
    algorithm_config.evaluation(evaluation_num_workers=3)
    algorithm = algorithm_config.build()

    policy_id = algorithm.get_policy()._Policy__policy_id
    selected_eval_worker_ids = algorithm.evaluation_workers.healthy_worker_ids()
    all_batches = []

    for _ in range(1):
        batches = algorithm.evaluation_workers.foreach_worker(
            func=lambda w: w.sample(),
            local_worker=False,
            remote_worker_ids=selected_eval_worker_ids,
            timeout_seconds=algorithm.config.evaluation_sample_timeout_s,
        )
        all_batches.extend(batches)

    total_batch = concat_samples(all_batches)
    writer.write(total_batch[policy_id])

    # reader = JsonReader('/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_dataset/prov_dataset/output-2023-12-06_15-00-35_worker-0_0.json')
    # batch = reader.next()
