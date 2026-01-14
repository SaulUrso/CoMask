import os
import random

import numpy as np
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)  # type: ignore
random.seed(0)


class_set = ["Call", "Hop", "typing", "Walk", "Wave"]
label = [0, 1, 2, 3, 4]

NUM_OF_CLASS = 5
DIMENSION_OF_FEATURE = 900
NUM_OF_TOTAL_USERS = 120


def load_data(user_id):

    coll_class = []
    coll_label = []

    total_class = 0

    for class_id in range(NUM_OF_CLASS):
        read_path = "./large_scale_HARBox/" + str(user_id) + "/" + str(class_set[class_id]) + "_train" + ".txt"

        if os.path.exists(read_path):
            temp_original_data = np.loadtxt(read_path)
            temp_reshape = temp_original_data.reshape(-1, 100, 10)
            temp_coll = temp_reshape[:, :, 1:10].reshape(-1, DIMENSION_OF_FEATURE)
            count_img = temp_coll.shape[0]
            temp_label = class_id * np.ones(count_img)

            # print(temp_original_data.shape)
            # print(temp_coll.shape)

            coll_class.extend(temp_coll)
            coll_label.extend(temp_label)

            total_class += 1

    coll_class = np.array(coll_class)
    coll_label = np.array(coll_label)

    # print(coll_class.shape)
    # print(coll_label.shape)

    return coll_class, coll_label, DIMENSION_OF_FEATURE, total_class


def generate_data(test_percent, x_coll, y_coll):
    x_train, x_test, y_train, y_test = train_test_split(x_coll, y_coll, test_size=test_percent, random_state=0)

    return x_train, x_test, y_train, y_test


def count_analysis(y):
    count_class = np.zeros(NUM_OF_CLASS)

    for class_id in range(NUM_OF_CLASS):
        count_class[class_id] = np.sum(y == class_id)

    return count_class
