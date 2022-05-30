import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from skimage import io
import cv2
import os
#from data_aug import *
#from box_util import *


class CornellDataset(Dataset):
    def __init__(self, path, images_set, transform=None):
        super(CornellDataset, self).__init__()
        self.dataset_path = path
        self.imgset = images_set
        # self.transform = A.Compose([transforms.ToTensor(),
        #                                      A.HorizontalFlip(p=0.5),
        #                                      A.RandomBrightnessContrast(p=0.2)], bbox_params=A.BboxParams(format='albumentations', 
        #                                      label_fields=['class_labels'])) 

        self.transform = transforms.Compose([transforms.ToTensor(), transform])
        # self.flip = RandomHorizontalFlip()
        # self.scale = RandomScale()
        # self.translate = RandomTranslate()

        self.classes = ('__background__',
                         'bin_01', 'bin_02', 'bin_03', 'bin_04', 'bin_05',
                         'bin_06', 'bin_07', 'bin_08', 'bin_09', 'bin_10',
                         'bin_11', 'bin_12', 'bin_13', 'bin_14', 'bin_15',
                         'bin_16', 'bin_17', 'bin_18', 'bin_19')
        self.empty_list = []
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self.height = 224
        self.width = 224
        rect_dataset = self.fetch_gt_rect()
        self.rect_dataset_bbox = rect_dataset[0]
    

    def __len__(self):
        return len(self.rect_dataset_bbox) 

    def __getitem__(self, index):

        #Note: Index being passes is different than the image name, hence this weird synchronoization
        image_path = self.get_image_path(self.img_idx[index])
        img = io.imread(image_path)
        #print(img.size())
        gt_img_rect = self.rect_dataset_bbox[index]
        gt_img_class = gt_img_rect[0]
        gt_img_box = gt_img_rect[1]
        
        #gt_img_class = np.sort(gt_img_class)
        #If the image has only one class
        #print(gt_img_class.size)
        if (gt_img_class.size == 1):
            gt_class = gt_img_class[0]
            gt_bbox = gt_img_box[0]

        else:
            classes, count = np.unique(gt_img_class, return_counts=True)
            if (classes[0] == 0) and (np.around(count[0]/count.sum())>np.around(1./count.size)):
                gt_class = gt_img_class[0]
                gt_bbox = gt_img_box[0]
            else:
                class_idx = np.argmax(count)
                gt_class = classes[class_idx]
                bbox_idx = np.where(gt_img_class == gt_class)[0]
                gt_bbox = gt_img_box[bbox_idx[0]]
        
        #Convert to tensors
        
         
        # img, gt_bbox = self.scale(img, gt_bbox.reshape(1, -1).astype(np.float64))
        # img, gt_bbox  = self.translate(img, gt_bbox.reshape(1, -1).astype(np.float64))
        # img, gt_bbox = self.flip(img, gt_bbox.reshape(1, -1).astype(np.float64))
        x1, y1, x2, y2 = gt_bbox/224
        gt_bbox=[(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
        img = self.transform(img.copy())
        gt_bbox = torch.tensor(gt_bbox)
        if gt_bbox.shape[0] == 1:
            gt_bbox = gt_bbox.squeeze(0)
        gt_class_bbox = [torch.tensor(gt_class), torch.tensor(gt_bbox)]

        return img, gt_class_bbox

    #Helper functions
    def get_image_path(self, index):
        filetype = ".png"
        path = os.path.join(self.dataset_path, "Images", index+filetype)
        return path

    def get_img_idx(self):
        file = os.path.join(self.dataset_path, "ImageSets", self.imgset+".txt")
        with open(file) as f:
            #Strip to remove the \n at the end of the line
            image_idx = [i.strip() for i in f.readlines()]
        return image_idx

    def get_coordinates(self, index):
        file = os.path.join(self.dataset_path, "Annotations", index+".txt")
        #print(index, file)

        if (os.stat(file).st_size) == 0:
            self.empty_list.append(index)
            #print('Empty files: {}'.format(file))
        else:
            with open(file) as f:
                file_content = f.readlines()
            
            #Storing bounding box coordinartes and class information
            gt_rect = np.zeros((len(file_content), 4), dtype=np.uint8)
            gt_class = np.zeros((len(file_content)), dtype=np.int32)

            #Loop through the data (line) and store
            i = 0
            for line in file_content:
                #Data is of format [class, x1, y1, x2, y2]
                line_content = line.split()
                #print(line_content)
                
                gt_class[i] = int(line_content[0])
                x1 = float(line_content[1])
                x2 = float(line_content[3])
                y1 = float(line_content[2])
                y2 = float(line_content[4])
                if ((x1 < 0) or (x1 > self.width) or (x2 < 0) or (x2 > self.width) or (y1 < 0) or (y1 > self.height) or (y2 < 0) or (y2 > self.height)):
                    continue
                gt_rect[i, : ] = [x1, y1, x2, y2]
                i+=1
            return gt_class, gt_rect

    def fetch_gt_rect(self):
        self.img_idx = self.get_img_idx()
        gt_rect = []
        data_len = len(self.img_idx)
        # for idx in range(data_len):
        #     #Each index has a tuple gt_rect[idx][0] = class and [1] = box coordinates
        #     gt_rect.append(self.get_coordinates(self.img_idx[idx]))
        gt_rect = [self.get_coordinates(image) for image in self.img_idx]

        for idx in self.empty_list:
            self.img_idx.remove(idx)
        for i in range(gt_rect.count(None)):
            gt_rect.remove(None)

        gt_rect_dataset = [gt_rect, self.img_idx]
        return gt_rect_dataset

#Test code for the class
if __name__ == "__main__":
    dataset_path = "dataset/cornell"
    img_set = "train"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    inv_normalize = transforms.Normalize(
                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                            std=[1/0.229, 1/0.224, 1/0.225])
    #Image to tensor conversion is made implicit inside the class
    dataset = CornellDataset(dataset_path, img_set, transform=normalize)
    img, gt_class_bbox = dataset.__getitem__(25)
    bbox = (gt_class_bbox[1]).numpy() * 224
    bbox = bbox.astype(np.uint8)
    print(f"bbox: {bbox}")
    
    img = np.transpose(img,(1,2,0)).numpy().astype(np.uint8).copy() 
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    cv2.imwrite("DatasetCheck.png", img)