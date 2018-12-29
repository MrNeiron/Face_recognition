import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
from Prepare_data import Get_data, Shuffle_data

np.random.seed(1)

RESOLUTION = (96,96)
GRAYSCALE = True
INPUT_SHAPE = (RESOLUTION[0], RESOLUTION[1], 1 if GRAYSCALE else 3)
NUM_EXAMPLES = 15#15
NUM_FOLDERS = 70#70
START_FOLDER = 200#150
BATCH_SIZE = 100#100
LEARNING_RATE = 1.#0.009
EPOCHS = 100#15

VALIDATION_SIZE = 0.2
RANDOM_STATE = 2018

X1_input, X2_input, Y_input = Get_data(path_p = "../../Datasets/Faces_dataset/Faces", 
                                       path_n = "../../Datasets/Faces_dataset/Faces",
                                      resolution = RESOLUTION,
                                      grayscale = GRAYSCALE,
                                      num_examples = NUM_EXAMPLES,
                                      num_folders = NUM_FOLDERS,
                                      input_shape = INPUT_SHAPE,
                                      start_folder = START_FOLDER)

X_train, X_test, Y_train, Y_test = Shuffle_data(X1_input, X2_input, Y_input, VALIDATION_SIZE, RANDOM_STATE)

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X1 = tf.placeholder(shape = [None, n_H0, n_W0, n_C0], dtype = tf.float32)
    X2 = tf.placeholder(shape = [None, n_H0, n_W0, n_C0], dtype = tf.float32)
    Y = tf.placeholder(shape = [None, n_y], dtype = tf.float32)
    
    return X1, X2, Y

def initialize_parameters(reuse = False):
    tf.set_random_seed(1)
    
    with tf.variable_scope("conv1", reuse = reuse):
        W1 = tf.get_variable("W1",
                             [4,4,1,6], 
                             initializer = tf.contrib.layers.xavier_initializer(seed=0))
    
    with tf.variable_scope("conv2", reuse = reuse):
        W2 = tf.get_variable("W2",
                             [10,10,6,64], 
                             initializer = tf.contrib.layers.xavier_initializer(seed=0))
        
    with tf.variable_scope("conv3", reuse = reuse):
        W3 = tf.get_variable("W3",
                             [7,7,64,128], 
                             initializer = tf.contrib.layers.xavier_initializer(seed=0))
        
    with tf.variable_scope("conv4", reuse = reuse):
        W4 = tf.get_variable("W4",
                             [4,4,128,128], 
                             initializer = tf.contrib.layers.xavier_initializer(seed=0))
        
    with tf.variable_scope("conv5", reuse = reuse):
        W5 = tf.get_variable("W5",
                             [7,7,128,256], 
                             initializer = tf.contrib.layers.xavier_initializer(seed=0))

    
    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4,
                  "W5": W5}
    
    return parameters

def forward_propagation(X, parameters, reuse = False):
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    W4 = parameters["W4"]
    W5 = parameters["W5"]
    
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1], padding="VALID")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1], strides = [1,8,8,1], padding = "VALID")
    
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1], padding = "SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize = [1,4,4,1], strides = [1,4,4,1], padding = "SAME")
    
    Z3 = tf.nn.conv2d(P2,W3, strides = [1,1,1,1], padding = "SAME")
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3,ksize = [1,4,4,1], strides = [1,4,4,1], padding = "SAME")
    
    Z4 = tf.nn.conv2d(P3,W4, strides = [1,1,1,1], padding = "SAME")
    A4 = tf.nn.relu(Z4)
    P4 = tf.nn.max_pool(A4,ksize = [1,4,4,1], strides = [1,4,4,1], padding = "SAME")
    
    Z5 = tf.nn.conv2d(P4,W5, strides = [1,1,1,1], padding = "SAME")
    A5 = tf.nn.relu(Z5)
    P5 = tf.nn.max_pool(A5,ksize = [1,4,4,1], strides = [1,4,4,1], padding = "SAME")
    
    P5 = tf.contrib.layers.flatten(P5)
    
    with tf.variable_scope("fc1", reuse=reuse):
        Z6 = tf.contrib.layers.fully_connected(P5, 10)
    
    return Z6

def forward_propagation_end(Z6_1, Z6_2):
    Sub = tf.subtract(Z6_1, Z6_2)
    
    F = tf.contrib.layers.fully_connected(Sub, 1, activation_fn = None)
    
    return F

def compute_cost(Z6, Y):
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z6, labels = Y))
    
    return cost

def train_model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
         num_epochs = 1, minibatch_size = 64, print_cost = True, load_model = True, load_model_path = ""):

    ops.reset_default_graph()
    tf.set_random_seed(13)
    seed = 13
    (m, n_H0, n_W0, n_C0) = X_train[0].shape
    n_y = Y_train.shape[1]
    costs = []
    
    X1, X2, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    
    parameters1 = initialize_parameters()
    parameters2 = initialize_parameters(reuse = True)
    
    Z6_1 = forward_propagation(X1, parameters1)
    Z6_2 = forward_propagation(X2, parameters2, reuse = True)
    Z6 = forward_propagation_end(Z6_1, Z6_2)

    cost = compute_cost(Z6, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        if (load_model == False): sess.run(init)
        else: saver.restore(sess, load_model_path)
        
        for epoch in range(num_epochs):
            
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed +1
            minibatches = random_mini_batches2(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                
                (minibatch_X, minibatch_Y) = minibatch
                
                _, temp_cost = sess.run([optimizer, cost], feed_dict = {X1:minibatch_X[0], X2:minibatch_X[1], Y:minibatch_Y})
                
                minibatch_cost += temp_cost /num_minibatches
            
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
                
        saver.save(sess, load_model_path)  

        a1_train = sess.run(Z6_1, {X1: X_train[0]})
        a2_train = sess.run(Z6_2, {X2: X_train[1]})

        dist = np.linalg.norm(a1_train - a2_train, axis = 1).reshape(Y_train.shape)
        
        sum_bool = np.count_nonzero((dist >= 0.5) == (Y_train >= 0.5))
        
        train_accuracy = (sum_bool / Y_train.shape[0])*100
        print("Train accuracy = ", train_accuracy)
        
        a1 = sess.run(Z6_1, {X1: X_test[0]})
        a2 = sess.run(Z6_2, {X2: X_test[1]})

        dist = np.linalg.norm(a1 - a2, axis = 1).reshape(Y_test.shape)
        
        sum_bool = np.count_nonzero((dist >= 0.5) == (Y_test >= 0.5))
        
        test_accuracy = (sum_bool / Y_test.shape[0])*100
        print("Test accuracy = ", test_accuracy)

    return train_accuracy, test_accuracy, parameters1, parameters2

train_acc, test_acc, parameters1, parameters2 = train_model(X_train, 
                                                            Y_train, 
                                                            X_test, 
                                                            Y_test,
                                                            learning_rate = LEARNING_RATE,
                                                            num_epochs = EPOCHS,
                                                            minibatch_size = BATCH_SIZE,
                                                            load_model = False,
                                                            load_model_path = "models/FaceModel.ckpt")


