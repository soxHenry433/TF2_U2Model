import tensorflow as tf
import json
from pycocotools import mask,coco
from functools import partial

def DeCompileCoCo(coco_path):
    varname = "myCOCO_soxhenry"
    cc = 0
    if varname in globals():
        varname = varname + str(cc)
        cc+=1
    print(f"\"{varname}\" will be global reserved variable")
    globals()[varname] = coco.COCO(coco_path)
    img_id_list = []
    img_path_list = []
    for i in globals()[varname].imgs:
        anno_id = globals()[varname].getAnnIds(imgIds=i)
        if len(anno_id) == 1:
            seg = globals()[varname].loadAnns(anno_id)[0]["segmentation"]
            if len(seg) < 1:
                continue
            img_path = globals()[varname].loadImgs(i)[0]["path"]
            
            img_id_list.append(i)
            img_path_list.append(img_path)
    
    return img_path_list, img_id_list, varname
            

def MakeDataSet(Path_list, Y_list, varname, BATCH = 32, SIZE = 512, Augment = True, SHUFFLE = True):
    Path = tf.data.Dataset.from_tensor_slices(Path_list)
    Y_list = tf.data.Dataset.from_tensor_slices(Y_list)

    _ReadResize, Read_COCO, _readjson = MakeFun(SIZE,varname)
    Path = Path.map(_ReadResize, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    Y = Y_list.map(Read_COCO, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    DataSet = tf.data.Dataset.zip((Path, Y))

    #DataSet.repeat() should be used in model.fit
    if SHUFFLE is True:
        DataSet = DataSet.shuffle(20)
    DataSet = DataSet.batch(BATCH,drop_remainder = True)
    DataSet = DataSet.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    return DataSet    


def MakeFun(SIZE,varname):

    def _readjson(i):
        i = int(i.numpy())
        anno_id = globals()[varname].getAnnIds(imgIds=i)
        h = globals()[varname].loadImgs(i)[0]["height"]
        w = globals()[varname].loadImgs(i)[0]["width"]
        seg = globals()[varname].loadAnns(anno_id)[0]["segmentation"]
        rles = mask.frPyObjects(seg,h,w)
        rle = mask.merge(rles)
        M = mask.decode(rle)
        return M,h,w
    
    @tf.function
    def _ReadResize(file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_png(image, channels = 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if tf.shape(image)[1] != SIZE or tf.shape(image)[0] != SIZE:
            image = tf.image.resize_with_pad(image, target_height = SIZE, target_width = SIZE)
        image = image / 0.5
        image = image - 1
        return image

    @tf.function
    def Read_COCO(i):
        image,h,w = tf.py_function(_readjson,[i],[tf.float32,tf.int32,tf.int32])
        image = tf.reshape(image,[h,w,1])
        if tf.shape(image)[1] != SIZE or tf.shape(image)[0] != SIZE:
            image = tf.image.resize_with_pad(image, target_height = SIZE, target_width = SIZE)
        return image

    return _ReadResize,Read_COCO,_readjson

if __name__ == "__main__":
    coco_path = "/mnt/d/Scaphoid/Seg_Code/Json/Val0904.json"
    img_path_list,img_id_list,varname = DeCompileCoCo(coco_path)
    DataSet = MakeDataSet(img_path_list,img_id_list, varname, SIZE = 256, BATCH = 16,SHUFFLE = False)
    
    for X,Y in DataSet:
        print(X.shape,Y.shape)
