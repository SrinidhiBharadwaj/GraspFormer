from audioop import tostereo
from fileinput import filename
import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from skimage import io

class OCIDDataset(Dataset):
    def __init__(self, path, image_set, transform=None):
        super(OCIDDataset, self).__init__()

        lut = {"train_set":"training_0", "val_set":"validation_0"}
        self.dataset_path = path
        self.image_set = lut[image_set] #Train, val, test
        self.transform = transforms.Compose([transforms.ToTensor(), transform]) 
        self.stored_path = self.parse_dataset()

    def __len__(self):
        return len(self.stored_path["images"])

    def __getitem__(self, index):
        #Get the image
        image_path, ann_path = self.get_imageandannotations_path(index)
        print(image_path)
        img = io.imread(image_path)
        img = self.transform(img)

        #Get the annotations
        gt_bbox = self.fetch_bbox_data(ann_path)
        print(gt_bbox)

    #Helper functions
    def fetch_bbox_data(self, ann_path):
        with open(ann_path) as f:
            temp_bbox = []
            boxes_list = []
            for line in f.readlines():
                line = line.rstrip()
                [x, y] = line.split(" ") #White space
                x = float(x)
                y = float(y)
                temp_bbox.append((x, y))

                if len(temp_bbox) == 4:#Every 2 lines
                    boxes_list.append(temp_bbox)
                    temp_bbox = []
        return boxes_list
    
    def get_annotation_path(self, seqpath):
        pass

    def parse_dataset(self):
        filepath = os.path.join(self.dataset_path, "data_split", self.image_set+".txt")
        file_path_list = [] #Tentative: There has gotta be a better method for doing this
        filename_list = []
        with open(filepath) as f:
            for line in f.readlines():
                file_dir = line.split(",")[0].strip()
                image_name = line.split(",")[1].strip()
                file_path_list.append(file_dir)
                filename_list.append(image_name)

        return {"path":file_path_list, "images":filename_list}

    def get_imageandannotations_path(self, index):
        seqpath = self.stored_path["path"][index]
        image_name = self.stored_path["images"][index]
        img_folder = "rgb"
        ann_folder = "Annotations"
        image_path = os.path.join(self.dataset_path, seqpath, img_folder, image_name)
        ann_path = os.path.join(self.dataset_path, seqpath, ann_folder, image_name[:-4]+".txt")
        return image_path, ann_path

if __name__ == "__main__":

    ocid_path = "dataset/ocid"
    image_set = "train_set"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ocid_dataset = OCIDDataset(ocid_path, image_set, transform=normalize)
    ocid_dataset.__getitem__(1)
