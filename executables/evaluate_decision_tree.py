import ray
from ray.rllib.algorithms import PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.policy.policy import PolicySpec

import models.register_model
import environments.environment_creator
from models.decision_tree import DecisionTree

if __name__ == '__main__':
    ray.init(local_mode=True)

    config = (
        PPOConfig()
        .environment(env='mutli_agent_pong')     # disable_env_checking=True , env_config={'frameskip': 1, 'full_action_space': False, 'repeat_action_probability': 0.0}
        .evaluation(evaluation_num_workers=1)
    )
    config.policies = {'decision_tree': PolicySpec(policy_class=DecisionTree, config={'decision_tree_path': '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/sklearn_tree_classifier/decision_tree_classifier.pkl'})}
    config.policy_mapping_fn = lambda agent_id, *args, **kwargs: 'decision_tree'
    config.policy_map_capacity = 1

    algo = config.build()

    print(algo.evaluate())
