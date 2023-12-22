from ray.rllib.policy import Policy

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from executables.approximate_policy_decision_trees import detection_moving_objects, display_image


class DecisionTree(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.decision_tree: DecisionTreeClassifier = joblib.load(config['decision_tree_path'])

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):

        # observations_moving_objects = []
        # for obs in obs_batch:
        #     observations_moving_objects.append(detection_moving_objects(obs))
        # observations_moving_objects = np.array(observations_moving_objects)

        display_image(obs_batch[0])

        observations_moving_objects = obs_batch
        observations_moving_objects = observations_moving_objects.reshape(observations_moving_objects.shape[0], -1)

        actions = self.decision_tree.predict(observations_moving_objects)
        # actions = np.array([self.action_space.sample()] * obs_batch.shape[0])

        return actions, [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
