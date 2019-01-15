from keras.models import Model, load_model
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape, Dropout
from keras import regularizers
from keras.engine.topology import Input
from keras.optimizers import Adam
from keras import backend as K
#import numpy as np

from Prepare_data import Get_data
from utils import to_file_params, TimeControll, data_generator

from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from json import dump as json_dump
from os.path import exists
from os import makedirs


#MAIN
TYPE = 16.1
LOAD_TYPE = 16.1
LOAD_MODEL = True

#OTHERS
DESCRIPTION = "Tiny model with some new data"
PRINT_LOGS = False
SHOW_FIGURES = True

#IMAGE SIZE
RESOLUTION = (96,96)
GRAYSCALE = True
INPUT_SHAPE = (RESOLUTION[0], RESOLUTION[1], 1 if GRAYSCALE else 3)

#IMAGES TRAIN
NUM_EXAMPLES = 25 if TYPE != 0 else 15
START_EXAMPLES = 0 if TYPE != 0 else 0

#IMAGES VALIDATION
NUM_EXAMPLES_VAL = 2 if TYPE != 0 else 2
START_EXAMPLES_VAL = NUM_EXAMPLES if TYPE != 0 else 0

#FOLDERS TRAIN
NUM_FOLDERS = 4680 if TYPE != 0 else 8
START_FOLDER = 0 if TYPE != 0 else 0
BATCH_SIZE_FOLDER = 5 if TYPE != 0 else 2

#FOLDERS VALIDATION
NUM_FOLDERS_VAL = NUM_FOLDERS if TYPE != 0 else 5
START_FOLDER_VAL = START_FOLDER if TYPE != 0 else 900
BATCH_SIZE_FOLDER_VAL = BATCH_SIZE_FOLDER if TYPE != 0 else 2


#TRAIN ITTERATION
#BATCH_SIZE = 80 if TYPE != 0 else 50
EPOCHS = 10 if TYPE != 0 else 5

#TRAIN
LEARNING_RATE = 0.001
L2 = 0.0007

#DIRECTIONS
NAME_HISTORY = f"history/FaceModel_whaleType/train/history{TYPE}.txt" if TYPE != 0 else f"history/FaceModel_whaleType/train/history_train_test{TYPE}.txt"
NAME_JSON_HISTORY = f"history/FaceModel_whaleType/JSON/history{TYPE}.json" if TYPE != 0 else f"history/FaceModel_whaleType/JSON/history_test{TYPE}.json"

NAME_FIGURE = f"figures/FaceModel_whaleType/train/{TYPE}/"


SAVE_MODEL_DIR = "models/FaceModel_whaleType"
SAVE_EACH_MODEL_DIR = f"{SAVE_MODEL_DIR}/{TYPE}"

SAVE_MODEL_NAME = f"{SAVE_MODEL_DIR}/Model_{TYPE}.h5" if TYPE != 0 else f"{SAVE_MODEL_DIR}/Model_test_{TYPE}.h5" 
SAVE_EACH_MODEL_NAME = (f"{SAVE_EACH_MODEL_DIR}/Model_" + "{epoch:02d}-{val_loss:.2f}.h5") if TYPE != 0 else (f"{SAVE_EACH_MODEL_DIR}/Model_test_" + "{epoch:02d}-{val_loss:.2f}.h5")

LOAD_MODEL_NAME = f"{SAVE_MODEL_DIR}/Model_{LOAD_TYPE}.h5" if TYPE != 0 else f"{SAVE_MODEL_DIR}/Model_test_{LOAD_TYPE}.h5" 


DATASET_P = DATASET_N = "../Datasets/Faces_dataset/Faces"

TRAIN_EXAMPLES = NUM_EXAMPLES*NUM_FOLDERS*2
TEST_EXAMPLES = NUM_EXAMPLES_VAL*NUM_FOLDERS_VAL*2
ALL_EXAMPLES = TRAIN_EXAMPLES+TEST_EXAMPLES
    
trainGen = data_generator(DATASET_P,
                          DATASET_N,
                          resolution = RESOLUTION,
                          grayscale = GRAYSCALE,
                          num_examples = NUM_EXAMPLES,
                          start_examples = START_EXAMPLES,
                          num_folders = NUM_FOLDERS,
                          start_folder = START_FOLDER,
                          batch_size_folder = BATCH_SIZE_FOLDER,
                          input_shape = INPUT_SHAPE,
                          print_logs = PRINT_LOGS)

valGen = data_generator(DATASET_P,
                        DATASET_N,
                        resolution = RESOLUTION,
                        grayscale = GRAYSCALE,
                        num_examples = NUM_EXAMPLES_VAL,
                        start_examples = START_EXAMPLES_VAL, 
                        num_folders = NUM_FOLDERS_VAL,
                        start_folder = START_FOLDER_VAL,
                        batch_size_folder = BATCH_SIZE_FOLDER_VAL,
                        input_shape = INPUT_SHAPE,
                        print_logs = PRINT_LOGS)

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

    inp = Input(shape=img_shape) # 96x96x1
    x   = Conv2D(64, (9,9), strides=2, activation='relu', **kwargs)(inp)

    x   = MaxPooling2D((2, 2), strides=(2, 2))(x) # 50x50x64
    for _ in range(1):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3,3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 25x25x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1,1), activation='relu', **kwargs)(x) # 25x25x128
    for _ in range(1): x = subblock(x, 64, **kwargs)
    #x = Dropout(0.5)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 12x12x384
    x = BatchNormalization()(x)
    x = Conv2D(256, (1,1), activation='relu', **kwargs)(x) # 12x12x512
    #for _ in range(1): x = subblock(x, 128, **kwargs)
    
    x             = GlobalMaxPooling2D()(x) # 384
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

if LOAD_MODEL:
    FaceModel = load_model(LOAD_MODEL_NAME)
else:
    FaceModel, _,_ = build_model(INPUT_SHAPE, lr = LEARNING_RATE, l2 = L2)


if LOAD_MODEL: to_file_params(NAME_HISTORY, [f"load_type = {LOAD_TYPE}"], False)
to_file_params(NAME_HISTORY, [f"input_shape = {INPUT_SHAPE}\nnum_examples = {NUM_EXAMPLES}\nnum_examples_val = {NUM_EXAMPLES_VAL}\nstart_examples = {START_EXAMPLES}\nstart_examples_val = {START_EXAMPLES_VAL}\nnum_folders = {NUM_FOLDERS}\nstart_folder = {START_FOLDER}\nbatch_size_folder = {BATCH_SIZE_FOLDER}\nnum_folders_val = {NUM_FOLDERS_VAL}\nstart_folder_val = {START_FOLDER_VAL}\nbatch_size_folder_val = {BATCH_SIZE_FOLDER_VAL}\nepochs = {EPOCHS}\nlearning_rate = {LEARNING_RATE}\nl2 = {L2}\nname_history = {NAME_HISTORY}\nsave_model_name = {SAVE_MODEL_NAME}\ndataset_positive = {DATASET_P}\ndataset_negative = {DATASET_N}\ndescription = {DESCRIPTION}\n"], with_lines = False)

to_file_params(NAME_HISTORY, [f"all_examples = {ALL_EXAMPLES}\ntrain_examples = {TRAIN_EXAMPLES}\ntest_examples = {TEST_EXAMPLES}\n"], with_lines = False)

if not exists(SAVE_EACH_MODEL_DIR):
    makedirs(SAVE_EACH_MODEL_DIR)

callbacks = ModelCheckpoint(SAVE_EACH_MODEL_NAME, 
                            monitor=["val_loss"], 
                            mode="min",
                            verbose = 1)
callback_list = [callbacks]

my_time = TimeControll()
time_start = my_time.get_start_time()
to_file_params(NAME_HISTORY, [f"\tStart time = {time_start[0]}:{time_start[1]}"], with_lines = False)

hist = FaceModel.fit_generator(trainGen, 
                               epochs = EPOCHS, 
                               steps_per_epoch = NUM_FOLDERS//BATCH_SIZE_FOLDER,
                               validation_data = valGen,
                               validation_steps = NUM_FOLDERS_VAL//BATCH_SIZE_FOLDER_VAL,
                               callbacks = callback_list,
                               verbose = 1)

my_time.set_end_time()

FaceModel.save(SAVE_MODEL_NAME)

time_end = my_time.get_end_time()
time_spend = my_time.get_spend_time()

to_file_params(NAME_HISTORY,[f"\tEnd time = {time_end[0]}:{time_end[1]}\n\tSpend time = {time_spend[0]}:{time_spend[1]}:{time_spend[2]}"], with_lines = False)

to_file_params(NAME_HISTORY,[f"\tAccuracy: {hist.history['acc'][-1]}, Val Accuracy: {hist.history['val_acc'][-1]}, Loss: {hist.history['loss'][-1]}, Val Loss: {hist.history['val_loss'][-1]}"], with_lines = False)


out = [f"Epoch {i+1}/{EPOCHS}\n\t\t loss: {loss}, acc: {acc}, val_loss: {val_loss}, val_acc: {val_acc}" for i,(loss,acc,val_loss,val_acc) in enumerate(zip(hist.history["loss"],hist.history["acc"],hist.history["val_loss"],hist.history["val_acc"]))]

to_file_params(NAME_HISTORY, out)

json_dump(hist.history, open(NAME_JSON_HISTORY, 'a'))

if SHOW_FIGURES:
    from utils import monitor_process

    monitor_process(hist.history, ["acc","val_acc","loss","val_loss"], NAME_FIGURE, TYPE)