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
NUM_OF_TOTAL_USERS = 116


def load_data(user_id):
    # dataset append and split

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


# node_info = np.zeros((116,4))

# for user_id in range(1,117):

# 	x_coll, y_coll, dimension, total_class = load_data(user_id)
# 	x_train,x_test,y_train,y_test = generate_data(0.6, x_coll, y_coll)

# 	print(x_train.shape)
# 	print(y_train.shape)
# 	print(x_test.shape)
# 	print(y_test.shape)
# 	print(y_test)

# 	node_info[user_id-1,0] = x_train.shape[0] + x_test.shape[0]
# 	node_info[user_id-1,1] = x_train.shape[0]
# 	node_info[user_id-1,2] = x_test.shape[0]
# 	node_info[user_id-1,3] = total_class

# np.savetxt("node_info.txt", node_info)

# NUM_OF_USERS = 9
# NUM_TRAIN_EXAMPLES_PER_USER = (50 * np.ones(NUM_OF_USERS)).astype(int)
# NUM_TEST_EXAMPLES_PER_USER = (40 * np.ones(NUM_OF_USERS)).astype(int)

# [x_coll, y_coll, D] = load_data()

# node_train_index= node_define(NUM_TRAIN_EXAMPLES_PER_USER, NUM_OF_USERS)
# node_test_index= node_define(NUM_TEST_EXAMPLES_PER_USER, NUM_OF_USERS)

# (x_train, y_train, x_test, y_test) = \
# generate_data(NUM_TRAIN_EXAMPLES_PER_USER, NUM_TEST_EXAMPLES_PER_USER, NUM_OF_USERS, node_train_index, node_test_index, x_coll, y_coll)

# count_train_class = count_analysis(y_train, node_train_index, NUM_OF_USERS)
# count_test_class = count_analysis(y_test, node_test_index, NUM_OF_USERS)
# print(count_train_class)
# print(count_test_class)


# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(y_test)


# NOTE: uncomment from here for original


# count_user_data = np.zeros(NUM_OF_TOTAL_USERS)


# # load data
# for current_user_id in range(1, NUM_OF_TOTAL_USERS + 1):
#     print("USER:", current_user_id)
#     x_coll, y_coll, dimension, num_of_class = load_data(current_user_id)
#     # print(count_analysis(y_coll))
#     count_user_data[current_user_id - 1] = y_coll.shape[0]

# print(np.min(count_user_data))
# print(np.max(count_user_data))
# print(np.mean(count_user_data))
# print(np.std(count_user_data))
