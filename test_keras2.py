from keras.models import load_model

from Prepare_data import Get_data, Shuffle_data
from utils import to_file_params, TimeControll

TYPE = 9.3
RESOLUTION = (96,96)
GRAYSCALE = True
INPUT_SHAPE = (RESOLUTION[0], RESOLUTION[1], 1 if GRAYSCALE else 3)
NUM_EXAMPLES = 15 if TYPE != 0 else 2
NUM_FOLDERS = 20 if TYPE != 0 else 2
START_FOLDER = 200 if TYPE != 0 else 0

NAME_HISTORY = f"history/history_test{TYPE}.txt"
LOAD_MODEL_NAME = f"models/FaceModel_whaleType{TYPE}.h5" if TYPE != 0 else f"models/FaceModel_train_test{TYPE}.h5"

VALIDATION_SIZE = 0.2
RANDOM_STATE = 2018

X1_input, X2_input, Y_input = Get_data(path_p = "../Datasets/Faces_dataset/Faces", 
                                       path_n = "../Datasets/Faces_dataset/Faces",
                                      resolution = RESOLUTION,
                                      grayscale = GRAYSCALE,
                                      num_examples = NUM_EXAMPLES,
                                      num_folders = NUM_FOLDERS,
                                      input_shape = INPUT_SHAPE,
                                      start_folder = START_FOLDER)

X_test = [X1_input, X2_input]
Y_test = Y_input 

to_file_params(NAME_HISTORY, [f"\ninput_shape = {INPUT_SHAPE}\nnum_examples = {NUM_EXAMPLES}\nnum_folders = {NUM_FOLDERS}\nstart_folder = {START_FOLDER}\nvalidation_size = {VALIDATION_SIZE}\nrandom_state = {RANDOM_STATE}\nname_history = {NAME_HISTORY}\nload_model_name = {LOAD_MODEL_NAME}\n"], False)

to_file_params(NAME_HISTORY, [f"test_examples = {Y_test.shape[0]}\n"], False)

FaceModel = load_model(LOAD_MODEL_NAME)

my_time = TimeControll()
score = FaceModel.evaluate(X_test, Y_test, verbose = 1)
my_time.set_end_time()


time_start = my_time.get_start_time()
time_end = my_time.get_end_time()
time_spend = my_time.get_spend_time()

out = [f"\tStart time={time_start[0]}:{time_start[1]}, End time= {time_end[0]}:{time_end[1]}, Spend time= {time_spend[0]}:{time_spend[1]}:{time_spend[2]}"]

out.append(f"\n\tTest accuracy = {score[-1]}")

print(f"Test accuracy: {score[-1]}")

to_file_params(NAME_HISTORY, out)
