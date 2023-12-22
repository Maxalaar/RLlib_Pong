import numpy as np

from executables.approximate_policy_decision_trees import get_size_dataset, load_data
from executables.generate_dataset_from_policy import save_data_in_data_frame

if __name__ == '__main__':
    datasets_path = './ray_datasets/data_1.h5'
    refine_datasets_path = './ray_datasets/refine_data_1'
    directory = './sklearn_tree_classifier/'
    model_filename = 'decision_tree_classifier.pkl'
    datasets_size = get_size_dataset(datasets_path)
    print('dataset size : ' + str(datasets_size))

    size_load_data = 150000
    start_index = 0
    stop_index = start_index + size_load_data
    # print(str(start_index) + '/' + str(datasets_size))
    # observations, actions = load_data(datasets_path, start_index=start_index, stop_index=stop_index)

    while stop_index <= datasets_size:
        print(str(start_index) + '/' + str(datasets_size))
        observations, actions = load_data(datasets_path, start_index=start_index, stop_index=stop_index)

        observations, unique_index = np.unique(observations, axis=0, return_index=True)
        actions = actions[unique_index]

        save_data_in_data_frame(path=refine_datasets_path, new_observations=observations, new_actions=actions)
        start_index += size_load_data
        stop_index += size_load_data

        # observations, unique_index = np.unique(observations, axis=0, return_index=True)
        # actions = actions[unique_index]
        #
        # start_index += size_load_data
        # stop_index += size_load_data
        # print(str(start_index) + '/' + str(datasets_size))
        #
        # new_observations, new_actions = load_data(datasets_path, start_index=start_index, stop_index=stop_index)
        #
        # observations = np.append(observations, new_observations)
        # actions = np.append(actions, new_actions)

    print(str(start_index) + '/' + str(datasets_size))
    observations, actions = load_data(datasets_path, start_index=start_index, stop_index=datasets_size)
    observations, unique_index = np.unique(observations, axis=0, return_index=True)
    actions = actions[unique_index]
    save_data_in_data_frame(path=refine_datasets_path, new_observations=observations, new_actions=actions)

    refine_datasets = get_size_dataset(refine_datasets_path)
    print('refine dataset size : ' + str(refine_datasets))
