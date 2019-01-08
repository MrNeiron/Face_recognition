from datetime import datetime
from keras.callbacks import Callback
from pandas import DataFrame
from Prepare_data import Get_data
from os.path import exists
from os import makedirs
import matplotlib.pyplot as plt


def data_generator(dir_p, dir_n, resolution, grayscale, num_examples, num_folders, input_shape, batch_size_folder, start_examples, start_folder, print_logs):
    while True:
        for current_start_folder in range(start_folder,start_folder+num_folders,batch_size_folder):
            X1_input, X2_input, Y1_input = Get_data(dir_p, 
                                                    dir_n,
                                                    resolution = resolution,
                                                    grayscale = grayscale,
                                                    num_examples = num_examples,
                                                    num_folders = batch_size_folder,
                                                    start_folder = current_start_folder,
                                                    start_examples = start_examples,
                                                    input_shape = input_shape,
                                                    print_logs = print_logs)

            yield [X1_input,X2_input],Y1_input
            
def monitor_process(history, metrics, figure_name, type_model):
    metrics_list = {m:history[m] for m in metrics}
    epochs = range(len(metrics_list[metrics[0]]))
    if not exists(figure_name):
        makedirs(figure_name)
    for metric,values in metrics_list.items():
        plt.plot(epochs, values)
        plt.title(f"Type: {type_model} - {metric}")
        if type_model != 0:
            plt.savefig(figure_name + f'{metric}-{values[-1]:.2f}.jpg', quality = 100, dpi = 150)
        else:
            plt.savefig(figure_name + f'test_{metric}-{values[-1]:.2f}.jpg', quality = 100, dpi = 150)
        plt.show()
        #plt.savefig("history/test_accuracy.jpg", bbox_inches = "tight", quality = 100, dpi = 150)#save_fig
            
def to_file_params(filename, params, with_lines = True):
    f = open(filename, "a")
    for k in params:
        f.write(k + '\n')
    if with_lines: f.write('\n-----------------------------------------\n\n')
    f.close()
    
def get_params_from_file(filename, name, with_name = False):
    filename = str(filename)
    f = open(filename)

    param = [line[line.index(name):-1] for line in f if name in line][0]    
    if not with_name:
        param = param.split(' = ')[1]
        try:
            if '(' in param:
                param = param.split(', ')
                param[0] = param[0][1:]
                param[-1] = param[-1][:-1]
                param = [int(p) for p in param]
                param = tuple(param)
            else:
                param = float(param) if '.' in param else int(param)
        except:
            param = param
    f.close()
    return param

def get_all_params(filename, with_name, print_params, *args):
    params = {param: get_params_from_file(filename, param, with_name = with_name) for param in args}
    if print_params:
        [print(value) for value in params.values()]
    return params

def comm_dict(params, names):    
    new_params = dict()
    for k in params[0].keys():
        new_params[k] = {name:param[k] for param,name in zip(params,names)}
        
    return new_params

def get_pandas_dataframe(d):
    return DataFrame.from_dict(d).T
    
def get_metrics_from_file(filename, name, with_name = False):
    filename = str(filename)
    f = open(filename)
    lines = [line for line in f if name in line]
    lines = [line[line.index(name):-1] for line in lines]
    lines = [line.split(' ') for line in lines]
    lines = [line[0:2] if '=' not in line[1] else [line[0],line[2]] for line in lines]
    lines = [[line[0], line[1] if line[1][-1] != ',' else line[1][:-1]] for line in lines]
    if not with_name:
        lines = [float(line[1]) for line in lines]
    else:
        lines = [line[0]+' '+line[1] for line in lines]
    f.close()
    return lines

class TimeControll():
    def __init__(self, start_time = datetime.now().time()):
        self.start_time = start_time
        self.end_time = start_time
        
    def get_start_time(self): return (self.start_time.hour, self.start_time.minute, self.start_time.second)
    def set_start_time(self): self.start_time = datetime.now().time()
        
    def get_end_time(self): return (self.end_time.hour, self.end_time.minute, self.end_time.second)
    def set_end_time(self): self.end_time = datetime.now().time()
    
    def get_spend_time(self, start = None, finish = None):
        
        if finish is None: finish = self.end_time
        if start is None: start = self.start_time
        
        add_hour = 0
        if finish.hour < start.hour: add_hour += 24
        s = abs((finish.hour + add_hour)*3600+finish.minute*60+finish.second - (start.hour*3600+start.minute*60+start.second))
        m = s //60
        s %= 60
        h = m //60
        m %= 60
        
        return h,m,s

class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.val_acc = []
        self.val_loss = []
        self.acc = []
        self.loss = []
        self.metrics = {"acc": self.acc,
                        "val_acc": self.val_acc,
                        "loss": self.loss,
                        "val_loss": self.val_loss}

    def on_epoch_end(self, batch, logs={}):
        self.val_acc.append(logs.get('val_acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))