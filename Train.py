import tensorflow as tf
import pandas as pd
from GenJson import DeCompileCoCo,MakeDataSet
from U2Net import U2NET
from Misc import MetricCollector


mirrored_strategy = tf.distribute.MirroredStrategy()
GLOBAL_BATCH_SIZE=8
epoch = 300
LR = 1e-4
SIZE = 256


img_path_list_train, img_id_list, varname = DeCompileCoCo("/mnt/d/Scaphoid/Seg_Code/Json/Train0904.json")
my_generator = MakeDataSet(img_path_list_train, img_id_list, varname, BATCH = GLOBAL_BATCH_SIZE, SIZE = SIZE, Augment = False, SHUFFLE = True)
my_generator = mirrored_strategy.experimental_distribute_dataset(my_generator)

img_path_list, img_id_list, varname = DeCompileCoCo("/mnt/d/Scaphoid/Seg_Code/Json/Val0904.json")
val_generator = MakeDataSet(img_path_list, img_id_list, varname, BATCH = GLOBAL_BATCH_SIZE, SIZE = SIZE, Augment = False, SHUFFLE = False)
val_generator = mirrored_strategy.experimental_distribute_dataset(val_generator)

with mirrored_strategy.scope():
    myModel = U2NET()

    my_optimizer = tf.keras.optimizers.Nadam(lr=LR) #tf.keras.optimizers.Adam(lr=args.learningrate)

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
        return tf.reduce_sum(loss0 / (SIZE*SIZE), axis = [1,2])

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


if __name__ == "__main__":
    Train = False
    Infer = True
    
    if Train:
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
            myVal_ACC = MC.GetResult("Val_IOU")
            
            cc += 1
            if myVal_ACC > bestmyValACC and j > 50:
                myModel.save_weights(f"/mnt/d/Scaphoid/Model_save/U2/0907_{j}_{myVal_ACC:.3f}_tf")
                bestmyValACC = myVal_ACC
                cc = 0
                
    if Infer:
        import cv2
        import os
        import numpy as np
        
        SaveDir = "/mnt/d/Scaphoid/Model_save/U2/Test"
        img_path_list, img_id_list, varname = DeCompileCoCo("/mnt/d/Scaphoid/Seg_Code/Json/Test0904.json")
        val_generator = MakeDataSet(img_path_list, img_id_list, varname, BATCH = 1, SIZE = SIZE, Augment = False, SHUFFLE = False)
        
        myModel.load_weights("/mnt/d/Scaphoid/Model_save/U2/0907_202_0.859_tf")
        
        for i,dataset_inputs in enumerate(val_generator):
            Raw, Y = dataset_inputs
            X,_,_,_,_,_,_ = myModel(Raw,training = False)
            X = tf.squeeze(X,axis=0)
            X = tf.cast(X>0.5,tf.float32)
            X = tf.image.grayscale_to_rgb(X)
            X = tf.image.convert_image_dtype(X, tf.uint8)
            X = X.numpy()
            
            Raw = tf.squeeze(Raw,axis=0)
            Raw = (Raw + 1 )*0.5
            Raw = tf.image.convert_image_dtype(Raw, tf.uint8)
            Raw = Raw.numpy()

            Y = tf.squeeze(Y,axis=0)
            Y = tf.cast(Y>0.5,tf.float32)
            Y = tf.image.grayscale_to_rgb(Y)
            Y = tf.image.convert_image_dtype(Y, tf.uint8)
            Y = Y.numpy()
            
            M = np.where(X>=100,Raw,0)
            upper_part = np.concatenate([Raw,M],axis = 1)
            lower_part = np.concatenate([Y,X],axis = 1)
            IMAGE = np.concatenate([upper_part,lower_part],axis = 0)
            
            OutPutName = f"{SaveDir}/{os.path.basename(img_path_list[i])}"
            print(OutPutName)
            cv2.imwrite(OutPutName,IMAGE)
