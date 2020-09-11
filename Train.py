import tensorflow as tf
import pandas as pd
from GenJson import DeCompileCoCo,MakeDataSet
from U2Net import U2NET
from Misc import MetricCollector
import os
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow_addons as tfa
import sys
import datetime
import json

import argparse
parser = argparse.ArgumentParser()
#Basic
parser.add_argument("-m", "--model", type=str, default = "Model", help = "Model")
parser.add_argument("-g", "--generator", type=str, default = "Generator", help = "Generator")
#Some extension
parser.add_argument("-M", "--Pretrained", type=str, default = None, help = "Pretrained")
parser.add_argument("-t", "--TestList", type=str, default = None, help = "Test List Path, eg. Test_list_{FoldNum}" )
parser.add_argument("-t2", "--TestList2", type=str, default = None, help = "Another test list" )
parser.add_argument("-T", "--TrainList", type=str, default = None, help = "Train List Path, eg. Train_list_{FoldNum}" )
parser.add_argument("-I", "--Infer", type=int, default = 0, help = "Number of image to inference" )
#Hyper parameter
parser.add_argument("-b", "--batch_size", type=int, default = 32, help = "batch_size")
parser.add_argument("-l", "--learningrate", type=float, default = 1e-4, help = "learningrate")
parser.add_argument("-s", "--Resize", type=int, default = 512, help = "Resize")
parser.add_argument("-O", "--Optimizer", type=str, default = None, help = "Optimizer eg. tf.keras.optimizers.Adam(lr=0.005)")
#Other extension
parser.add_argument("-v", "--Verbose", type=int, default = 0, help = "Verbose")
parser.add_argument("-e", "--epoch", type=int, default = 75, help = "epoch")
#Output option
parser.add_argument("-N", "--Note", type=str, default = None, help = "Note")
parser.add_argument("-B", "--BaseDir", type=int, default = 1, help = "Add BaseDir to path or not")
parser.add_argument("-o", "--output", type=str, default = None, help = "Output")
parser.add_argument("-D", "--OutPutDir", type=str, default = None, help = "OutPutDir")
parser.add_argument("-mp", "--MixedPrecision", type=int, default = 0, help = "MixedPrecision")
parser.add_argument("-xla", "--EnableXLA", type=int, default = 1, help = "EnableXLA")
args = parser.parse_args()

#Configure GPU 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)
if len(physical_devices) > 1:
    Multi_GPU = True

#XLA
if bool(args.EnableXLA):
    tf.config.optimizer.set_jit(True)
    
#Test for mixed precion
if bool(args.MixedPrecision):
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

#Import package
Generator = __import__(args.generator, fromlist = [args.generator])
Model = __import__(args.model, fromlist = [args.model])
import Misc as Misc

#Hyperparameter
mirrored_strategy = tf.distribute.MirroredStrategy()
GLOBAL_BATCH_SIZE = args.batch_size
epoch = args.epoch
LR = args.learningrate
SIZE = args.Resize

#Config Ouput dir and name
SaveName = args.output
if args.output is None:
    SaveName = "LR" + str(args.learningrate)  + "B" + str(args.batch_size) + "S" + str(args.Resize)
    if args.Note is not None:
        SaveName = SaveName + "_" + str(args.Note)
BASE_DIR = Misc.GetBASE_DIR()
OutPutDir = args.OutPutDir
if args.OutPutDir is None:
    OutPutDir = args.model
SavePath=f'{BASE_DIR}/Model_save/{OutPutDir}'
if not os.path.exists(SavePath):
    os.mkdir(SavePath)
LOGDIR_profiler = f"file://{SavePath}/{SaveName}.log"


img_path_list_train, img_id_list, varname = DeCompileCoCo(args.TrainList)
my_generator = MakeDataSet(img_path_list_train, img_id_list, varname, BATCH = GLOBAL_BATCH_SIZE, SIZE = SIZE, Augment = False, SHUFFLE = True)
my_generator = mirrored_strategy.experimental_distribute_dataset(my_generator)

img_path_list, img_id_list, varname = DeCompileCoCo(args.TestList)
val_generator = MakeDataSet(img_path_list, img_id_list, varname, BATCH = GLOBAL_BATCH_SIZE, SIZE = SIZE, Augment = False, SHUFFLE = False)
val_generator = mirrored_strategy.experimental_distribute_dataset(val_generator)

with mirrored_strategy.scope():
    myModel = Model.U2NET()
    if args.Pretrained is not None:
        myModel.load_weights(args.Pretrained)
    
    if args.Optimizer is not None:
        my_optimizer = eval(args.Optimizer)
    else:
        my_optimizer = tf.keras.optimizers.Adam(lr=LR)

    bce_loss0 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    Loss = tf.keras.metrics.Mean()
    ValLoss = tf.keras.metrics.Mean()
    IOU = tf.keras.metrics.MeanIoU(num_classes=2)
    Val_IOU = tf.keras.metrics.MeanIoU(num_classes=2)

    MC = MetricCollector(
        metric_list = {
            "Loss": Loss,
            "IOU": IOU,
        }, metric_list_val = {
            "ValLoss": ValLoss,
            "Val_IOU": Val_IOU,
        },)

    def muti_bce_loss_fusion(d0,labels_v):
        loss0 = bce_loss0(d0,labels_v)
        return tf.reduce_sum(loss0 , axis = [1,2])/ (SIZE*SIZE)

    def Cal_loss(d0,Y):
        per_example_loss = muti_bce_loss_fusion(d0,Y)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    def myOneStep(dataset_inputs):
        X, Y = dataset_inputs
        
        with tf.GradientTape() as disc_tape:
            d0, _, _, _, _, _, _ = myModel(X,training=True)
            loss = Cal_loss(d0, Y)
        
        gradients_of_model = disc_tape.gradient(loss, myModel.trainable_variables)
        my_optimizer.apply_gradients(zip(gradients_of_model, myModel.trainable_variables))
        
        MC.metric_list["Loss"].update_state(loss)
        MC.metric_list["IOU"].update_state(d0,Y)

    def myInfer(dataset_inputs):
        X, Y = dataset_inputs
        d0, _, _, _, _, _, _ = myModel(X,training=False)
        loss = Cal_loss(d0, Y)
        MC.metric_list_val["ValLoss"].update_state(loss )
        MC.metric_list_val["Val_IOU"].update_state(d0,Y)

    
@tf.function
def distributed_train_step(dataset_inputs):
    return mirrored_strategy.run(myOneStep,args=(dataset_inputs,))

@tf.function
def distributed_val_step(dataset_inputs):
    return mirrored_strategy.run(myInfer,args=(dataset_inputs,))

with open(f"{SavePath}/{SaveName}.json",'w') as FF:
    output_Dict = vars(args)
    output_Dict.update({"Date": str(datetime.datetime.now()),
                "SavePath": SavePath, 
                "SaveName": SaveName,
                'RawCommand': ' '.join(sys.argv),
                })
    json.dump(output_Dict, FF, indent = 4, separators = (',',': '))


if __name__ == "__main__":    
    cc = 0
    bestmyValACC = 0.
    for j in range(1,epoch+1):
        pb_i = tf.keras.utils.Progbar(len(img_path_list_train))

        for dataset_inputs in my_generator:
            distributed_train_step(dataset_inputs)
            pb_i.add(GLOBAL_BATCH_SIZE)

        for dataset_inputs in val_generator:
            distributed_val_step(dataset_inputs)
        
        MC.Result()
        MC.PrintToScreen(j)
        MC.WriteToFile(j,LOGDIR_profiler)
        myVal_ACC = MC.GetResult("Val_IOU")
        
        cc += 1
        if myVal_ACC > bestmyValACC and j > 50:
            save_weight_path = f"{SavePath}/{SaveName}_{j}_{myVal_ACC:.3f}_tf"
            myModel.save_weights(save_weight_path)
            bestmyValACC = myVal_ACC
            cc = 0
    
       
    if args.Infer > 0:
        import cv2
        import os
        import numpy as np
        
        if not os.path.exists(f"{SavePath}/Infer/"):
            os.makedirs(f"{SavePath}/Infer/")
        
        if args.TestList2 is None:
            args.TestList2 = args.TestList
        img_path_list, img_id_list, varname = DeCompileCoCo(args.TestList2)
        val_generator = MakeDataSet(img_path_list, img_id_list, varname, BATCH = 1, SIZE = SIZE, Augment = False, SHUFFLE = False)
                
        myModel.load_weights(save_weight_path)
        
        test_IOU = tf.keras.metrics.MeanIoU(num_classes=2)
        
        for i,dataset_inputs in enumerate(val_generator):
            Raw, Y = dataset_inputs
            X,_,_,_,_,_,_ = myModel(Raw,training = False)
            test_IOU.update_state(X,Y)
            
            if i < args.Infer:
                X = tf.squeeze(X,axis=0)
                X = tf.cast(X>0.5,tf.float32)
                X = tf.image.grayscale_to_rgb(X)
                X = tf.image.convert_image_dtype(X, tf.uint8)
                X = X.numpy()
                
                Raw = tf.squeeze(Raw,axis=0)
                Raw = (Raw + 1 )*0.5
                Raw = tf.image.convert_image_dtype(Raw, tf.uint8)
                Raw = Raw.numpy()[...,[2,1,0]]

                Y = tf.squeeze(Y,axis=0)
                Y = tf.cast(Y>0.5,tf.float32)
                Y = tf.image.grayscale_to_rgb(Y)
                Y = tf.image.convert_image_dtype(Y, tf.uint8)
                Y = Y.numpy()
                
                M = np.where(X>=100,Raw,0)
                upper_part = np.concatenate([Raw,M],axis = 1)
                lower_part = np.concatenate([Y,X],axis = 1)
                IMAGE = np.concatenate([upper_part,lower_part],axis = 0)
                
                OutPutName = f"{SavePath}/Infer/{os.path.basename(img_path_list[i])}"
                cv2.imwrite(OutPutName,IMAGE)
        
        iou = test_IOU.result().numpy()
        print(iou)
        
        with open(f"{SavePath}/{SaveName}.stats",'w') as FF:
            FF.write(f"IOU: {iou}\n")



