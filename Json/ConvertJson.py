import numpy as np
import json
import cv2
import pandas as pd
import copy
import random

class myCoCoObj():

    def __init__(self):
        self.img_path = []
        self.annotations = []
        self.meta = []
        self.ShapeList = []
        self.FinalJson = {}
        self.seg_list = []
        
        
    def AddOneLine(self, Points, file_path, SHAPE,seg, MetaData = None, check = True):
        if check:
            Points = self.CheckPoints(Points)
        if isinstance(Points[0],list):
            Points = Points[0]
        if len(Points) == 4:
            self.annotations.append(Points)
            self.img_path.append(file_path)
            self.meta.append(MetaData)
            self.ShapeList.append(SHAPE[:2])
            self.seg_list.append(seg)

    
    def Finalize(self, Category = None):
        image_path_list, image_idx_list = np.unique(np.array(self.img_path), return_inverse = True)
        Image_dict = []
        Annot_dict = []
        complete_img_idx = []
        for anno_idx, img_idx in enumerate(image_idx_list):
            
            if img_idx not in complete_img_idx:
                SingleImageDict = self.CreateSingleImageDict(image_idx = img_idx,
                                                             file_path = self.img_path[anno_idx],
                                                             SHAPE = self.ShapeList[anno_idx], 
                                                             MetaData = self.meta[anno_idx]
                                                             )
                Image_dict.append(SingleImageDict)
                complete_img_idx.append(img_idx)
            
            
            SingleAnnotDict = self.CreateSingleAnnotDict(anno_idx = anno_idx,
                                       img_idx = img_idx,
                                       bbox = self.annotations[anno_idx],
                                       seg = self.seg_list[anno_idx])
            Annot_dict.append(SingleAnnotDict)
        
        if Category is None:
            Category = [{
                'id': 1,
                'name': 'Scaphoid',
                'supercategory': ''
            }]
        self.FinalJson['images'] = Image_dict
        self.FinalJson['categories'] = Category
        self.FinalJson['annotations'] = Annot_dict
        print(f"There are {len(Image_dict)} images")
        print(f"There are {len(Category)} categories")
        print(f"There are {len(Annot_dict)} annotations")

    def SaveJson(self, SaveName):
        with open(SaveName,"w") as FF:
            json.dump(self.FinalJson,FF,indent = 4, separators = (',',': '))


    def Visualize(self):
        annotation = self.FinalJson['annotations'][-1]
        ii = annotation["image_id"]
        
        imgid_to_posid = {}
        for i in range(len(self.FinalJson['images'])):
            imgid_to_posid[self.FinalJson['images'][i]["id"]] = i
        img_path = self.FinalJson['images'][imgid_to_posid[ii]]["path"]
        bbox = annotation["bbox"]

        #cv2.namedWindow(img_path,cv2.WINDOW_NORMAL)
        #cv2.resizeWindow(img_path, (600,600))
        img = cv2.imread(img_path)
        cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,0,255), 3)
        cv2.imwrite("Test.png", img)
        #cv2.imshow(img_path, img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()


    def FromJson(self, JsonFile):
        with open(JsonFile,"r") as FF:
            Json = json.loads(FF.read())

        imgid_to_posid = {}
        for i in range(len(Json['images'])):
            imgid_to_posid[Json['images'][i]["id"]] = i

        for k in range(len(Json['annotations'])):
            Points = Json["annotations"][k]["bbox"]
            i = Json["annotations"][k]["image_id"]
            seg = Json['annotations'][k]["segmentation"]
            file_path = Json['images'][imgid_to_posid[i]]["path"]
            width = Json['images'][imgid_to_posid[i]]["width"]
            height = Json['images'][imgid_to_posid[i]]["height"]
            metadata = Json['images'][imgid_to_posid[i]]["metadata"]
            self.AddOneLine(Points = Points, file_path = file_path, SHAPE = (height, width), MetaData = metadata, check = False,seg=seg)

    def MergeCoCoObj(self, COCO):
        print(f"There are {len(self.annotations)} annotation in original coco")
        print(f"There are {len(COCO.annotations)} annotation in to-be-add coco")
        self.annotations.extend(COCO.annotations)
        self.img_path.extend(COCO.img_path)
        self.meta.extend(COCO.meta)
        self.ShapeList.extend(COCO.ShapeList)
        self.seg_list.extend(COCO.seg_list)
        print(f"There are {len(self.annotations)} annotation in merged coco")


    def Extract(self, file_list):
        AA = pd.Series(self.img_path)
        
        target_idx = AA[AA.isin(file_list)].index
        old_annotations = self.annotations
        old_img_path = self.img_path
        old_meta = self.meta
        old_ShapeList = self.ShapeList
        old_seglist = self.seg_list

        self.img_path = []
        self.annotations = []
        self.meta = []
        self.ShapeList = []
        self.seg_list = []
        for i in target_idx:
            self.annotations.append(old_annotations[i])
            self.img_path.append(old_img_path[i])
            self.meta.append(old_meta[i])
            self.ShapeList.append(old_ShapeList[i])
            self.seg_list.append(old_seglist[i])
        print(f"Extract {len(target_idx)} annotations")
        

    @staticmethod
    def CreateSingleImageDict(image_idx, file_path, SHAPE, MetaData):
        DD = {}
        DD['id'] = int(image_idx)
        DD['path'] = file_path
        DD['width'] = int(SHAPE[1])
        DD['height']= int(SHAPE[0])
        DD['file_name'] = file_path
        DD['metadata'] = MetaData
        return DD


    @staticmethod
    def CreateSingleAnnotDict(anno_idx, img_idx, bbox, seg = [],class_idx = 1):
        DD = {}
        DD['id'] = int(anno_idx)
        DD['image_id'] = int(img_idx)
        DD['category_id'] = int(class_idx)
        DD["bbox"] = bbox
        DD['iscrowd'] = False
        DD['segmentation'] = seg
        DD['area'] = int(bbox[2]*bbox[3])
        return DD


    @staticmethod
    def CheckPoints(Points):
        
        cX1 = int(Points[0]); cY1 = int(Points[1])
        cX2 = int(Points[2]); cY2 = int(Points[3])

        diff_x = cX2 - cX1
        diff_y = cY2 - cY1

        if diff_x > 0 and diff_y > 0:
            bbox = [ cX1, cY1, (cX2-cX1), (cY2-cY1)]
        elif diff_x <= 0 and diff_y <= 0:
            bbox = [ cX2, cY2, abs(diff_x), abs(diff_y)]
        elif diff_x <= 0 and diff_y > 0:
            cX3 = cX1 - abs(diff_x); cY3 = cY1
            bbox = [ cX3, cY3, abs(diff_x), abs(diff_y)]
        elif diff_x > 0 and diff_y <= 0:
            cX3 = cX2 - abs(diff_x); cY3 = cY2
            bbox = [ cX3, cY3, abs(diff_x), abs(diff_y)]

        return bbox


if __name__ == "__main__":
    DA = pd.read_csv("Data_info_JSON")
    uniq_id = DA.ID.unique()
    random.shuffle(uniq_id)
    ii = len(uniq_id) // 7
    DA_val = DA.loc[DA.ID.isin(uniq_id[:ii])]
    DA_test = DA.loc[DA.ID.isin(uniq_id[ii*6:])]
    DA_train = DA.loc[DA.ID.isin(uniq_id[ii:ii*6])]
    
    DA_val.loc["divide"] = "val"
    DA_test.loc["divide"] = "test"
    DA_train.loc["divide"] = "train"

    mycoco = myCoCoObj()
    mycoco.FromJson("All0904.json")
    
    test_coco = copy.deepcopy(mycoco)
    test_coco.Extract(DA_test.Path)
    test_coco.Finalize()
    test_coco.SaveJson("Test0904.json")

    test_coco = copy.deepcopy(mycoco)
    test_coco.Extract(DA_val.Path)
    test_coco.Finalize()
    test_coco.SaveJson("Val0904.json")
    
    test_coco = copy.deepcopy(mycoco)
    test_coco.Extract(DA_train.Path)
    test_coco.Finalize()
    test_coco.SaveJson("Train0904.json")

    DA = pd.concat([DA_val,DA_test,DA_train])
    
    DA.to_csv("Data_info_JSON0904",index=False)