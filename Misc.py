import pandas as pd
import numpy as np
import sys
import os
import tensorflow as tf




        

def readInfo(TRAIN_Y_PATH, part_val = None, Dummy = 1, return_id = False, SHUFFLE = True, BaseDir = True, BATCH = 0):
    DA = pd.read_csv(TRAIN_Y_PATH)
    if BATCH > 0:
        DA2 = DA.sample(n = BATCH - (len(DA.index) % BATCH))
        DA = pd.concat([DA,DA2]).reset_index()

    if SHUFFLE:
        DA = DA.sample(frac = 1).reset_index()
    file_list = DA['Path'].values
    if BaseDir:
        file_list = [ f'{BASE_DIR}/{i}' for i in file_list ]
    
    y = DA['Y'].astype(int).values
    if Dummy > 1:
        y = tf.one_hot(y, depth = Dummy, dtype = tf.float32)
    

    if part_val is not None:
        id_for_val = round(part_val * y.shape[0])
        index = np.arange(y.shape[0])
        np.random.shuffle(index)
        y = y[index,]
        file_list = file_list[index]

        y_val = y[:id_for_val,]
        file_val = file_list[:id_for_val]
        y_train = y[id_for_val:,]
        file_train = file_list[id_for_val:]
        return file_train,y_train, file_val,y_val

    if return_id is True:
        ID = DA['ID'].values
        return file_list, y, ID 

    return file_list,y


class MetricCollector():

    def __init__(self, metric_list = {}, metric_list_val = {}):
        self.metric_list = metric_list
        self.metric_list_val = metric_list_val
        self.result = {}

    def append(self, fun_dict, val = False):
        if val:
            self.metric_list_val.update(fun_dict)
        else:
            self.metric_list.update(fun_dict)

    def Reset(self):
        if len(self.metric_list_val) > 0:
            for i in self.metric_list_val:
                self.metric_list_val[i].reset_states()    

        if len(self.metric_list) > 0:
            for i in self.metric_list:
                self.metric_list[i].reset_states()    
    def Result(self):
        if len(self.metric_list) > 0:
            for i,metric in self.metric_list.items():
                self.result.update({i: metric.result().numpy()})
        if len(self.metric_list_val) > 0:
            for i,metric in self.metric_list_val.items():
                self.result.update({i: metric.result().numpy()})

    def GetResult(self, Target):
        if Target is not None:
            if Target in self.metric_list:
                return self.result[Target]
            if Target in self.metric_list_val:
                return self.result[Target]
        
        return None

            

    def PrintToScreen(self, epoch):
        print(f"epoch: {epoch:03d}", end="; ")
        if len(self.metric_list) > 0:
            for i in self.metric_list:
                print(i, end=": ")
                print(f"{self.result[i]:.3f}", end="; ")
        if len(self.metric_list_val) > 0:
            for i in self.metric_list_val:
                print(i, end=": ")
                print(f"{self.result[i]:.3f}", end="; ")
        print("", end = "\n")

    def WriteToFile(self, epoch, LOGDIR):
        if epoch==1:
            ss_list = [f"epoch"]        
            if len(self.metric_list) > 0:
                for i in self.metric_list:
                    ss_list.append(i)
            if len(self.metric_list_val) > 0:
                for i in self.metric_list_val:
                    ss_list.append(i)
            ff = ','.join(ss_list)
            tf.print(ff, output_stream = LOGDIR)

        ss_list = [f"{epoch:03d}"]        
        if len(self.metric_list) > 0:
            for i in self.metric_list:
                ss_list.append(f"{self.result[i]:.3f}")
        if len(self.metric_list_val) > 0:
            for i in self.metric_list_val:
                ss_list.append(f"{self.result[i]:.3f}")
        ff = ','.join(ss_list)
        tf.print(ff, output_stream = LOGDIR)


def GetBASE_DIR():
    BASE_DIR=os.path.dirname(os.path.abspath(__file__))
    return os.path.realpath(BASE_DIR+'/../')
    
BASE_DIR = GetBASE_DIR()
