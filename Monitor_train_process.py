import matplotlib.pyplot as plt
from utils import get_params_from_file as get_params, get_metrics_from_file as get_metrics, get_all_params, comm_dict, get_pandas_dataframe
import os

def set_bounds(y_bound, x_bound, y_step = None, x_step = None):
    plt.ylim(y_bound)
    plt.xlim(x_bound)
    if y_step is not None: plt.yticks(np.arange(y_bound[0],y_bound[1],y_step))
    if x_step is not None: plt.xticks(np.arange(x_bound[0],x_bound[1],x_step))
        

val_loss = list()
path = r"models\FaceModel_whaleType\16.1_2"
for file in os.listdir(path):
    val_loss.append(float(file[-7:-3]))
val_loss = [val_loss]
title = ["16.1"]

start = 0#len(val_loss)-3
finish = len(val_loss)
[plt.plot(a, label = t) for a,t in zip(val_loss[start:finish],title[start:finish])]
plt.ylabel("validation loss")
plt.xlabel("epochs")
plt.title("Validation loss")
plt.legend(bbox_to_anchor=(0., 1.12, 1., .102), loc=3,          ncol=2, mode="expand", borderaxespad=0.)

#set_bounds(y_bound = (0.75,0.86), x_bound = (60,90), y_step = 0.01, x_step = 3)

plt.savefig(f'figures/FaceModel_whaleType/train/{title[0]}/all_epochs/val_loss_{title[0]}-{len(val_loss[0])}.jpg', quality = 100, dpi = 150)

plt.show()