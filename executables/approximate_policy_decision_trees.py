import h5py
import cv2
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def load_data(path, number_steps=None, random=False):
    with h5py.File(path, 'r') as hf:
        if number_steps is not None and random == True:
            total_values = hf['observations'].shape[0]
            print('chose index')
            random_indices = np.random.choice(total_values, number_steps, replace=False)
            random_indices.sort()
            observations = hf['observations'][random_indices]
            actions = hf['actions'][random_indices]
        elif number_steps is not None and random == False:
            observations = hf['observations'][:number_steps]
            actions = hf['actions'][:number_steps]
        else:
            observations = hf['observations'][:]
            actions = hf['actions'][:]
        return observations, actions


def get_size_dataset(path) -> int:
    with h5py.File(path, 'r') as hf:
        return hf['observations'].shape[0]


def detection_moving_objects(image):
    # diff1 = cv2.absdiff(image[1], image[0])
    # diff2 = cv2.absdiff(image[2], image[1])
    diff3 = cv2.absdiff(image[3], image[2])

    # diff3 = cv2.absdiff((image[1] + image[0])/2, (image[2] + image[3])/2)

    # merged_diff = cv2.bitwise_or(cv2.bitwise_or(diff1, diff2), diff3)
    merged_diff = diff3

    threshold = 30
    _, motion_mask = cv2.threshold(merged_diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # image_to_draw = np.copy(image[3])
    number_box = 8
    moving_objects_box = np.full((number_box, 4), -1)

    for j in range(number_box):
        if j < len(contours):
            contour = contours[j]
            x, y, w, h = cv2.boundingRect(contour)
            moving_objects_box[j][0] = x
            moving_objects_box[j][1] = y
            moving_objects_box[j][2] = w
            moving_objects_box[j][3] = h

            # cv2.rectangle(image_to_draw, (x, y), (x + w, y + h), (0, 0, 0), 1)

            # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(image_to_draw, (x, y), (x + w, y + h), (0, 0, 0), 1)

    # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    # cv2.imshow('Image', image_to_draw)
    # cv2.waitKey(50)
    # cv2.destroyAllWindows()

    return moving_objects_box


def learning(X_train, X_test, y_train, y_test, max_depth=None, min_samples_leaf=1, min_samples_split=2):
    decision_tree_classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                      min_samples_split=min_samples_split)
    decision_tree_classifier.fit(X_train, y_train)

    y_pred = decision_tree_classifier.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    print("Précision du modèle sur les donné d'entrainement:", accuracy)

    y_pred = decision_tree_classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Précision du modèle sur les données de validation:", accuracy)
    return decision_tree_classifier


def find_sequence_indices(tableau, sequence):
    indices = []
    for i in range(tableau.shape[0]):
        for j in range(tableau.shape[1] - len(sequence) + 1):
            if np.array_equal(tableau[i, j:j + len(sequence)], sequence):
                indices.append((i, j))
    return indices


if __name__ == '__main__':
    datasets_path = './ray_datasets/data_1.h5'
    directory = './sklearn_tree_classifier/'
    model_filename = 'decision_tree_classifier.pkl'
    print('dataset size : ' + str(get_size_dataset(datasets_path)))

    # for i in range(1000):
    observations, actions = load_data(datasets_path)
    print('data load')
    observations = observations.transpose(0, 3, 1, 2)

    # observations = observations.reshape(observations.shape[0], -1)

    observations_moving_objects = []
    for i in range(observations.shape[0]):
        observations_moving_objects.append(detection_moving_objects(observations[i]))
    observations_moving_objects = np.array(observations_moving_objects)
    observations_moving_objects = observations_moving_objects.reshape(observations_moving_objects.shape[0], -1)
    print('processes obs')

    X_train, X_test, y_train, y_test = train_test_split(observations_moving_objects, actions, test_size=0.2)

    decision_tree_classifier = learning(X_train, X_test, y_train, y_test)

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    joblib.dump(decision_tree_classifier, os.path.join(directory, model_filename))
