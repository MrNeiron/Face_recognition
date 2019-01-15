from keras.models import load_model
import numpy as np

from Prepare_data import Get_data, Shuffle_data
from utils import to_file_params, TimeControll


TYPE = 16.1
E = 4
VALUE = 0.64

RESOLUTION = (96,96)
GRAYSCALE = True
INPUT_SHAPE = (RESOLUTION[0], RESOLUTION[1], 1 if GRAYSCALE else 3)
NUM_EXAMPLES = 2 if TYPE != 0 else 2
START_EXAMPLES = 25
NUM_FOLDERS = 8 if TYPE != 0 else 2
START_FOLDER = 600 if TYPE != 0 else 0

NAME_HISTORY = f"history/FaceModel_whaleType/predict/history_predict{TYPE}.txt"
LOAD_MODEL_NAME = f"models/FaceModel_whaleType/Model_{TYPE}.h5" if TYPE != 0 else f"models/FaceModel_whaleType/Model_test{TYPE}.h5"
#LOAD_MODEL_NAME = f"models/FaceModel_whaleType/{TYPE}/Model_0{E}-{VALUE}.h5"

IMAGE_SAVE_DIR = f"test_images/{TYPE}"

VALIDATION_SIZE = 0.2
RANDOM_STATE = 2018

X1_input, X2_input, Y_test = Get_data(path_p = "../Datasets/Faces_dataset/Faces", 
                                       path_n = "../Datasets/Faces_dataset/Faces",
                                      resolution = RESOLUTION,
                                      grayscale = GRAYSCALE,
                                      num_examples = NUM_EXAMPLES,
start_examples = START_EXAMPLES,
                                      num_folders = NUM_FOLDERS,
                                      input_shape = INPUT_SHAPE,
                                      start_folder = START_FOLDER)

X_test = [X1_input, X2_input]


to_file_params(NAME_HISTORY, [f"\ninput_shape = {INPUT_SHAPE}\nnum_examples = {NUM_EXAMPLES}\nstart_examples = {START_EXAMPLES}\nnum_folders = {NUM_FOLDERS}\nstart_folder = {START_FOLDER}\nvalidation_size = {VALIDATION_SIZE}\nrandom_state = {RANDOM_STATE}\nname_history = {NAME_HISTORY}\nload_model_name = {LOAD_MODEL_NAME}\n"], False)

to_file_params(NAME_HISTORY, [f"predict_examples = {Y_test.shape[0]}\n"], False)


FaceModel = load_model(LOAD_MODEL_NAME)

my_time = TimeControll()
score = FaceModel.predict(X_test, verbose = 1)
my_time.set_end_time()

score_r = np.round(score)

from cv2 import imwrite
from os.path import exists
from os import makedirs

if not exists(IMAGE_SAVE_DIR): 
    makedirs(IMAGE_SAVE_DIR) 

for i in range(Y_test.shape[0]):
    imwrite(IMAGE_SAVE_DIR + f"/test_image{i}.1_{Y_test[i]}-{score_r[i]}.jpg", X_test[0][i])
    imwrite(IMAGE_SAVE_DIR + f"/test_image{i}.2_{Y_test[i]}-{score_r[i]}.jpg", X_test[1][i])
    #print(f"Y{i} = {Y_train[i]}")

time_start = my_time.get_start_time()
time_end = my_time.get_end_time()
time_spend = my_time.get_spend_time()

out = [f"\tStart time={time_start[0]}:{time_start[1]}, End time= {time_end[0]}:{time_end[1]}, Spend time= {time_spend[0]}:{time_spend[1]}:{time_spend[2]}"]

scores = [f"{i})score: {s} - {t}: {s==t}" for i,(s, t) in enumerate(zip(score_r, Y_test))]
[print(s) for s in scores]

predict_accuracy = (np.count_nonzero(score_r==Y_test)/len(score_r))* 100

print("Predict accuracy = ", predict_accuracy)

out.append(f"\n\tPredict accuracy = {predict_accuracy}\n")


[out.append(s) for s in scores]

to_file_params(NAME_HISTORY, out)
