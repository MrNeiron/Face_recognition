from keras.models import load_model

from Prepare_data import Get_data, Shuffle_data
from utils import to_file_params, TimeControll, get_all_params




TYPE = 90

params = get_all_params(f"history/history{TYPE}.txt", False, False, 
                        "batch_size", 
                        "epochs", 
                        "learning_rate", 
                        "l2")

RESOLUTION = (96,96)
GRAYSCALE = True
INPUT_SHAPE = (RESOLUTION[0], RESOLUTION[1], 1 if GRAYSCALE else 3)
NUM_EXAMPLES = 15 if TYPE != 0 else 2
NUM_FOLDERS = 20 if TYPE != 0 else 2
START_FOLDER = 100 if TYPE != 0 else 0

NAME_HISTORY = f"history/history_test{TYPE}.txt"
LOAD_MODEL_NAME = f"models/FaceModel_whaleType{TYPE}.h5" if TYPE != 0 else f"models/FaceModel_train_test{TYPE}.h5"
DATASET_P = DATASET_N = "../Datasets/Faces_dataset/Faces"

BATCH_SIZE = params["batch_size"]
EPOCHS = params["epochs"]


LEARNING_RATE = params["learning_rate"]
L2 = params["l2"]

X1_input, X2_input, Y_input = Get_data(path_p = DATASET_P, 
                                       path_n = DATASET_N,
                                      resolution = RESOLUTION,
                                      grayscale = GRAYSCALE,
                                      num_examples = NUM_EXAMPLES,
                                      num_folders = NUM_FOLDERS,
                                      input_shape = INPUT_SHAPE,
                                      start_folder = START_FOLDER)

X_test = [X1_input, X2_input]
Y_test = Y_input 

to_file_params(NAME_HISTORY, [f"\ninput_shape = {INPUT_SHAPE}\nnum_examples = {NUM_EXAMPLES}\nnum_folders = {NUM_FOLDERS}\nstart_folder = {START_FOLDER}\nname_history = {NAME_HISTORY}\nload_model_name = {LOAD_MODEL_NAME}\ndataset_p = {DATASET_P}\ndataset_n = {DATASET_N}\n"], False)

to_file_params(NAME_HISTORY, [f"test_examples = {Y_test.shape[0]}\n"], False)

FaceModel = load_model(LOAD_MODEL_NAME)

my_time = TimeControll()
time_start = my_time.get_start_time()
to_file_params(NAME_HISTORY, [f"\tStart time = {time_start[0]}:{time_start[1]}"], with_lines = False)

score = FaceModel.evaluate(X_test, Y_test, verbose = 1)
my_time.set_end_time()


time_end = my_time.get_end_time()
time_spend = my_time.get_spend_time()

to_file_params(NAME_HISTORY,[f"\tEnd time = {time_end[0]}:{time_end[1]}\n\tSpend time = {time_spend[0]}:{time_spend[1]}:{time_spend[2]}"], with_lines = False)

to_file_params(NAME_HISTORY,[f"\n\tTest accuracy = {score[-1]}"])

print(f"Test accuracy: {score[-1]}")