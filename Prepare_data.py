from Preprocess_image import take_n_resize_images as take
from sklearn.model_selection import train_test_split
import numpy as np


def PullOutArray(arr):
    newArr = np.zeros((arr.shape[0]*arr.shape[1], arr.shape[2], arr.shape[3], arr.shape[4]))

    counter = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            newArr[counter] = arr[i][j]
            counter += 1
    print(f"new_shape = {newArr.shape[0]}, {newArr.shape[1]}, {newArr.shape[2]}, {newArr.shape[3]}")
    return newArr

def Get_data(path_p, path_n, resolution, grayscale, num_examples, num_folders, input_shape, start_folder = 0):
    X1_input_p0 = np.zeros((num_folders, num_examples, input_shape[0], input_shape[1], input_shape[2]))
    X2_input_p0 = np.zeros((num_folders, num_examples, input_shape[0], input_shape[1], input_shape[2]))

    for i,j in enumerate(range(start_folder, num_folders+start_folder)):
        X1_input_p0[i] = take(f"{path_p}/({j})",
                             image_size = resolution,
                             grayscale = grayscale,
                             num_examples = num_examples)
        X2_input_p0[i] = X1_input_p0[i][::-1]
    
    print(X1_input_p0.shape)
    print(X2_input_p0.shape)

    X1_input_p = PullOutArray(X1_input_p0)
    X2_input_p = PullOutArray(X2_input_p0)

    Y_input_p = np.ones((X1_input_p.shape[0],1))
    print(Y_input_p.shape)



    X1_input_n0 = np.zeros((num_folders, num_examples, input_shape[0], input_shape[1], input_shape[2]))
    X2_input_n0 = np.zeros((num_folders, num_examples, input_shape[0], input_shape[1], input_shape[2]))

    for i,j in enumerate(range(start_folder, num_folders+start_folder)):
        X1_input_n0[i] = take(f"{path_n}/({j})",
                       image_size = resolution,
                       grayscale = grayscale,
                       num_examples = num_examples)
        X2_input_n0[i] = take(f"{path_n}/({j+1})",
                   image_size = resolution,
                   grayscale = grayscale,
                   num_examples = num_examples)

    print(X1_input_n0.shape)
    print(X2_input_n0.shape)

    X1_input_n = PullOutArray(X1_input_n0)
    X2_input_n = PullOutArray(X2_input_n0)

    Y_input_n = np.zeros((X1_input_n.shape[0],1))
    print("Y_input_n.shape: ", Y_input_n.shape)

    X1_input = np.vstack((X1_input_p, X1_input_n))
    print("X1_input.shape: ", X1_input.shape)

    X2_input = np.vstack((X2_input_p, X2_input_n))
    print("X2_input.shape: ", X2_input.shape)

    Y_input = np.vstack((Y_input_p, Y_input_n))
    print("Y_input.shape: ", Y_input.shape)
    
    return X1_input, X2_input, Y_input

def Shuffle_data(X1_input, X2_input, Y_input, validation_size, random_state):
    X1_train, X1_val, Y_train, Y_val = train_test_split(X1_input, Y_input, test_size = validation_size, random_state = random_state)
    X2_train, X2_val, Y_train, Y_val = train_test_split(X2_input, Y_input, test_size = validation_size, random_state = random_state)
    
    X_train = [X1_train, X2_train]
    X_val = [X1_val, X2_val]
    print("X_train.shape: ",X1_train.shape)
    print("X_val.shape: ", X1_val.shape)
    print("Y_train.shape: ", Y_train.shape)
    print("Y_val.shape: ", Y_val.shape)
    
    return X_train, X_val, Y_train, Y_val
