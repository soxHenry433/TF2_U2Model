import tensorflow as tf
import pandas as pd
from U2Net import U2NET
from skimage.measure import label, regionprops
import cv2
import os
import numpy as np

DATA_INFO = "/mnt/e/Glaucoma/Data/List/TP0818/Data_info_0818"
Model_weight = "/mnt/e/Glaucoma/Model_save/U2/0910/GenJsonLR0.0001B8S256_100_0.848_tf"
OutPutDir="/mnt/e/Glaucoma/Data/U2Crop"
SIZE = 256
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    myModel = U2NET()


@tf.function
def MyInfer(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels = 3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    X = tf.cast(image, tf.float32)
    
    if tf.shape(image)[1] != SIZE or tf.shape(image)[0] != SIZE:
        X = tf.image.resize_with_pad(X, target_height = SIZE, target_width = SIZE)
    X = X / 0.5
    X = X - 1

    Y_, _, _, _, _, _, _ = myModel(tf.expand_dims(X,axis = 0),training = False)
    Y_ = tf.cast(Y_>0.5,tf.float32)
    Y_ = tf.squeeze(Y_,axis=[0,-1])
        
    return Y_, tf.image.convert_image_dtype(image, tf.uint8)


if __name__ == "__main__":
    myModel.load_weights(Model_weight)
    
    DA = pd.read_csv(DATA_INFO)
    file_list = DA.Path.values
    
    Seg_path_list = []
    Crop_path_list = []
    FileName_list = []
    for i,filename in enumerate(file_list):
        Y_, Raw = MyInfer(tf.convert_to_tensor(filename))
        Y_ = Y_.numpy()
        Raw = Raw.numpy()[...,[2,1,0]]
        
        regions = regionprops(label(Y_))
        
        h, w, _ = Raw.shape
        if h >= w:
            ratio_ = Raw.shape[0] / SIZE
            pad = SIZE - Raw.shape[1] / ratio_
        else: 
            ratio_ = Raw.shape[1] / SIZE
            pad = SIZE - Raw.shape[0] / ratio_
            
        pad = pad * ratio_ /2
        for i in range(len(regions)):
            
            bbox = regions[i].bbox
            if regions[i].area/(SIZE**2) > 0.01 and len(bbox)==4:
                
                Raw2 = np.copy(Raw)
                if h < w:
                    x0 = max(0,int(bbox[0]*ratio_ - pad))
                    x1 = min(Raw.shape[0]-1,int(bbox[2]*ratio_ - pad))
                    Raw3 = Raw[x0:x1,int(bbox[1]*ratio_):int(bbox[3]*ratio_),:]
                    Raw2[x0:x1,int(bbox[1]*ratio_):int(bbox[3]*ratio_),:] = 0
                else:
                    x0 = max(0,int(bbox[1]*ratio_ - pad))
                    x1 = min(Raw.shape[1]-1,int(bbox[3]*ratio_ - pad))
                    Raw3 = Raw[int(bbox[0]*ratio_ ):int(bbox[2]*ratio_ ),x0:x1,:]
                    Raw2[int(bbox[0]*ratio_ ):int(bbox[2]*ratio_ ),x0:x1,:] = 0
                    
                try:
                    start = int(Raw2.shape[1]*0.2)
                    end = int(Raw2.shape[1] - Raw2.shape[1]*0.2)
                    Raw2 = Raw2[:,start:end,:]
                    Raw2 = tf.image.resize_with_pad(Raw2,512,512)
                    Seg_path = f"{OutPutDir}/Seg/Seg_{i}_{os.path.basename(filename)}"
                    Seg_path_list.append(Seg_path)
                    cv2.imwrite(Seg_path,Raw2.numpy())
                    
                    Crop_path = f"{OutPutDir}/Crop/Crop_{i}_{os.path.basename(filename)}"
                    Crop_path_list.append(Seg_path)
                    cv2.imwrite(Crop_path,Raw3)
                    
                    FileName_list.append(filename)
                    
                except:
                    print(f"{OutPutDir}/Seg_{i}_{os.path.basename(filename)}")
                    
    DA2 = pd.DataFrame({
        "Path": FileName_list,
        "seg_Path": Seg_path_list,
        "crop_Path": Crop_path_list
    })
    
    DA3 = DA2.merge(DA,on = "Path")
    DA3.to_csv("/mnt/e/Glaucoma/Data/List/TP0818/Data_info_seg",index=False)
        
'''
if h < w:
    x0 = max(0,int(bbox[0]*ratio_ - pad))
    x1 = min(Raw.shape[0]-1,int(bbox[2]*ratio_ - pad))
    Raw2 = Raw[x0:x1,int(bbox[1]*ratio_):int(bbox[3]*ratio_),:]
else:
    x0 = max(0,int(bbox[1]*ratio_ - pad))
    x1 = min(Raw.shape[1]-1,int(bbox[3]*ratio_ - pad))
    Raw2 = Raw[int(bbox[0]*ratio_ ):int(bbox[2]*ratio_ ),x0:x1,:]
'''