#LOCAL
from keras.models import Model
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras import regularizers
from keras.engine.topology import Input
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

from Prepare_data import Get_data, Shuffle_data
from utils import to_file_params, TimeControll, AccuracyHistory

TYPE = 11.1
DESCRIPTION = "Turning num epochs"
RESOLUTION = (96,96)
GRAYSCALE = True
INPUT_SHAPE = (RESOLUTION[0], RESOLUTION[1], 1 if GRAYSCALE else 3)
NUM_EXAMPLES = 15 if TYPE != 0 else 2
NUM_FOLDERS = 150 if TYPE != 0 else 2
START_FOLDER = 500 if TYPE != 0 else 0
BATCH_SIZE = 400 if TYPE != 0 else 1
EPOCHS = 50 if TYPE != 0 else 1

LEARNING_RATE = 0.001
L2 = 0.0007

VALIDATION_SIZE = 0.2
RANDOM_STATE = 2018

NAME_HISTORY = f"history/history{TYPE}.txt"
SAVE_MODEL_NAME = f"models/FaceModel_whaleType{TYPE}.h5" if TYPE != 0 else f"models/FaceModel_train_test{TYPE}.h5" 

DATASET_P = DATASET_N = "../Datasets/Faces_dataset/Faces"

X1_input, X2_input, Y_input = Get_data(path_p = DATASET_P, 
                                       path_n = DATASET_N,
                                      resolution = RESOLUTION,
                                      grayscale = GRAYSCALE,
                                      num_examples = NUM_EXAMPLES,
                                      num_folders = NUM_FOLDERS,
                                      input_shape = INPUT_SHAPE,
                                      start_folder = START_FOLDER)

X_train, X_val, Y_train, Y_val = Shuffle_data(X1_input, X2_input, Y_input, VALIDATION_SIZE, RANDOM_STATE)

def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y) # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y) # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y) # no activation # Restore the number of original features
    y = Add()([x,y]) # Add the bypass connection
    y = Activation('relu')(y)
    return y

def build_model(img_shape, lr, l2, activation='sigmoid'):

    ##############
    # BRANCH MODEL
    ##############
    regul  = regularizers.l2(l2)
    optim  = Adam(lr=lr)
    kwargs = {'padding':'same', 'kernel_regularizer':regul}

    inp = Input(shape=img_shape) # 384x384x1
    x   = Conv2D(64, (9,9), strides=2, activation='relu', **kwargs)(inp)

    x   = MaxPooling2D((2, 2), strides=(2, 2))(x) # 96x96x64
    for _ in range(1):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3,3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 48x48x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1,1), activation='relu', **kwargs)(x) # 48x48x128
    for _ in range(2): x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 24x24x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1,1), activation='relu', **kwargs)(x) # 24x24x256
    for _ in range(2): x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 12x12x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1,1), activation='relu', **kwargs)(x) # 12x12x384
    for _ in range(2): x = subblock(x, 96, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 6x6x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1,1), activation='relu', **kwargs)(x) # 6x6x512
    for _ in range(2): x = subblock(x, 128, **kwargs)
    
    x             = GlobalMaxPooling2D()(x) # 512
    branch_model  = Model(inp, x)
    
    ############
    # HEAD MODEL
    ############
    mid        = 32
    xa_inp     = Input(shape=branch_model.output_shape[1:])
    xb_inp     = Input(shape=branch_model.output_shape[1:])
    x1         = Lambda(lambda x : x[0]*x[1])([xa_inp, xb_inp])
    x2         = Lambda(lambda x : x[0] + x[1])([xa_inp, xb_inp])
    x3         = Lambda(lambda x : K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4         = Lambda(lambda x : K.square(x))(x3)
    x          = Concatenate()([x1, x2, x3, x4])

    x          = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x          = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x          = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x          = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x          = Flatten(name='flatten')(x)
    
    # Weighted sum implemented as a Dense layer.
    x          = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a      = Input(shape=img_shape)
    img_b      = Input(shape=img_shape)
    xa         = branch_model(img_a)
    xb         = branch_model(img_b)
    x          = head_model([xa, xb])
    model      = Model([img_a, img_b], x)
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model


FaceModel, _,_ = build_model(INPUT_SHAPE, lr = LEARNING_RATE, l2 = L2)


        
history = AccuracyHistory()

to_file_params(NAME_HISTORY, [f"input_shape = {INPUT_SHAPE}\nnum_examples = {NUM_EXAMPLES}\nnum_folders = {NUM_FOLDERS}\nstart_folder = {START_FOLDER}\nbatch_size = {BATCH_SIZE}\nepochs = {EPOCHS}\nlearning_rate = {LEARNING_RATE}\nl2 = {L2}\nvalidation_size = {VALIDATION_SIZE}\nrandom_state = {RANDOM_STATE}\nname_history = {NAME_HISTORY}\nsave_model_name = {SAVE_MODEL_NAME}\ndataset_positive = {DATASET_P}\ndataset_negative = {DATASET_N}\ndescription = {DESCRIPTION}\n"], with_lines = False)

to_file_params(NAME_HISTORY, [f"all_examples = {Y_input.shape[0]}\ntrain_examples = {Y_train.shape[0]}\ntest_examples = {Y_val.shape[0]}\n"], with_lines = False)

my_time = TimeControll()
time_start = my_time.get_start_time()
to_file_params(NAME_HISTORY, [f"\tStart time = {time_start[0]}:{time_start[1]}"], with_lines = False)



FaceModel.fit(X_train,Y_train,
                   batch_size = BATCH_SIZE,
                   epochs = EPOCHS,
                   verbose = 1,
                   validation_data = (X_val, Y_val),
                   callbacks = [history])
my_time.set_end_time()

FaceModel.save(SAVE_MODEL_NAME)

time_end = my_time.get_end_time()
time_spend = my_time.get_spend_time()

to_file_params(NAME_HISTORY,[f"\tEnd time = {time_end[0]}:{time_end[1]}\n\tSpend time = {time_spend[0]}:{time_spend[1]}:{time_spend[2]}"], with_lines = False)

to_file_params(NAME_HISTORY,[f"\tAccuracy: {history.acc[-1]}, Val Accuracy: {history.val_acc[-1]}, Loss: {history.loss[-1]}, Val Loss: {history.val_loss[-1]}"], with_lines = False)

out = [f"Epoch {i+1}/{len(history.acc)}\n\t\t loss: {loss}, acc: {acc}, val_loss: {val_loss}, val_acc: {val_acc}" for i,(loss,acc,val_loss,val_acc) in enumerate(zip(history.loss,history.acc,history.val_loss,history.val_acc))]

to_file_params(NAME_HISTORY, out)