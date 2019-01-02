from datetime import datetime
from keras.callbacks import Callback



def to_file_params(filename, params, with_lines = True):
    f = open(filename, "a")
    for k in params:
        f.write(k + '\n')
    if with_lines: f.write('\n-----------------------------------------\n\n')
    f.close()

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

    def on_epoch_end(self, batch, logs={}):
        self.val_acc.append(logs.get('val_acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))