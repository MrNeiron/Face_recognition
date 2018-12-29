from keras.models import load_model
from Prepare_data import Get_data, Shuffle_data


RESOLUTION = (96,96)
GRAYSCALE = True
INPUT_SHAPE = (RESOLUTION[0], RESOLUTION[1], 1 if GRAYSCALE else 3)
NUM_EXAMPLES = 15
NUM_FOLDERS = 20
BATCH_SIZE = 50
EPOCHS = 7
VALIDATION_SIZE = 0.2
RANDOM_STATE = 2018

X1_input, X2_input, Y_input = Get_data(path_p = "../../Datasets/Faces_dataset/Faces", 
                                       path_n = "../../Datasets/Faces_dataset/Faces",
                                      resolution = RESOLUTION,
                                      grayscale = GRAYSCALE,
                                      num_examples = NUM_EXAMPLES,
                                      num_folders = NUM_FOLDERS,
                                      input_shape = INPUT_SHAPE,
                                      start_folder = 40)

X_train, X_val, Y_train, Y_val = Shuffle_data(X1_input, X2_input, Y_input, VALIDATION_SIZE, RANDOM_STATE)


FaceModel = load_model("models/FaceModel.h5")

FaceModel.fit(X_train,Y_train,
                   batch_size = BATCH_SIZE,
                   epochs = EPOCHS,
                   verbose = 1,
                   validation_data = (X_val, Y_val))

FaceModel.save("models/FaceModel2.h5")
