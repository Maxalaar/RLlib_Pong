from ray.rllib.offline.json_reader import JsonReader
import h5py
import matplotlib.pyplot as plt

def load_partial_data(path, start_idx, end_idx):
    with h5py.File(path, 'r') as hf:
        observations = hf['observations'][start_idx:end_idx]
        actions = hf['actions'][start_idx:end_idx]
        return observations, actions

if __name__ == '__main__':
    # reader = JsonReader('/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_datasets/dataset_1/output-2023-12-06_16-26-45_worker-0_0.json')
    # batch = reader.next()

    # plt.imshow(batch['obs'][0])
    # plt.axis('off')  # Pour ne pas afficher les axes
    # plt.show()

    observations, actions = load_partial_data(
        '/ray_datasets/data_1.h5',
        0,
        1000,
    )

    fig = plt.figure()

    for i in range(len(observations)):
        plt.imshow(observations[i])
        plt.pause(0.005)
        plt.draw()

    plt.show()

    # for i in range(2):
    #     plt.imshow(observations[i])
    #     plt.axis('off')
    #     plt.show()

