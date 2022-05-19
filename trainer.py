import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from model import  DETR
from cornell_dataset import CornellDataset
from hungarian import HungarianMatcher
import os
from torch.utils.data import Dataset, DataLoader
###################################################################################################
import random
import math
from datetime import datetime
from collections import Counter

import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

def read_image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int16)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def create_bb_array(x):
    """Generates bounding box array from a train_df row"""
    return np.array([x[5],x[4],x[7],x[6]])

def resize_image_bb(read_path,write_path,bb,sz):
    """Resize an image and its bounding box and write image to new path"""
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1*sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1*sz), sz))
    new_path = str(write_path/read_path.parts[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)


def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

# random crop to the original size
def random_crop(x, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    if y:
        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def random_cropXY(x, Y, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY

def transformsXY(path, bb, transforms):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    Y = create_mask(bb, x)
    if transforms:
        rdeg = (np.random.random()-.50)*20
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5: 
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y)



##################################################################################################



class Trainer():
    def __init__(self, model, train_loader, val_loader, device, optimizer, loss_funtion, 
                 epochs=10, lr=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.loss_obj = loss_funtion
        self.optimizer = optimizer
       
        self.save_every = 10 #Save the model every n epochs

    def train_network(self, orientation_only=False):
        print("Beginning training!!")
        print(f"Number of training images: {len(self.train_loader)}")
        for epoch in range(self.epochs):
            running_orientation_loss = 0
            for t, (x, y_class, y_bb) in enumerate(self.train_loader):

                x = x.to(torch.float32).to(self.device)

                # class_label = y[0].long().to(self.device)
                bbox_label = y_bb.float().to(self.device)
                #print(bbox_label)
                
                bbox_pred, class_pred = self.model(x)
                
                
                #Learn orientation model weights regardless
                target_dic = {'bbox':bbox_label,'class':y_class.to(torch.long).to(device)}
                output_dic = {'bbox':bbox_pred,'class':class_pred}
                if epoch%5==0:
                    loss = self.loss_obj(target_dic,output_dic,verbose=True)
                else:   
                    loss = self.loss_obj(target_dic,output_dic)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_orientation_loss += loss.item()
            #print(f"Epoch: {epoch}, loss: {np.mean(running_orientation_loss)}")
            
            with torch.no_grad():
                running_loss_val = 0
                self.model.eval()
                for i, (x_val, y_class_val, y_bb_val) in enumerate(self.val_loader):
                    x_val = x_val.to(torch.float32).to(device)
                    bbox_label = y_bb_val.float().to(self.device)

                    bbox_pred, class_pred = self.model(x_val)
                    target_dic = {'bbox':bbox_label,'class':y_class_val.to(torch.long).to(device)}
                    output_dic = {'bbox':bbox_pred,'class':class_pred}
                    loss_val = self.loss_obj(target_dic,output_dic)
                    running_loss_val += loss_val.item()
                    
                print("Epoch: {0}, Training loss: {1}, \
                                Validation Loss: {2}".format(epoch,np.mean(running_orientation_loss),np.mean(running_loss_val)))


            # #print(f"Epoch: {epoch}/{self.epochs} --> Orientation_loss: {running_orientation_loss/len(self.train_loader)} \
            #  #                         Bbox Matching loss: TBF")
            
            if (epoch+1) % self.save_every==0:
                save_name = os.path.join('model_{}.ckpt'.format(epoch))
                torch.save({
                'epoch': epoch + 1,
                'model': self.model.state_dict(), # 'model' should be 'model_state_dict'
                'optimizer_bbox': self.optimizer.state_dict(),
                'loss': loss.item(),
                }, save_name) 
        
        print("Finished training!")

if __name__ == "__main__":
    df_train = pd.read_csv('traffic_data.csv')
    X = df_train[['new_path', 'new_bb']]
    Y = df_train['class']

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    def normalize(im):
        """Normalizes images with Imagenet stats."""
        imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        return (im - imagenet_stats[0])/imagenet_stats[1]

    class RoadDataset(Dataset):
        def __init__(self, paths, bb, y, transforms=False):
            self.transforms = transforms
            self.paths = paths.values
            self.bb = bb.values
            self.y = y.values
        def __len__(self):
            return len(self.paths)

        def box_xyxy_to_cxcywh(self,np_bb):
            x0,y0,x1,y1 = np_bb
            b = [(x0 + x1) / 2, (y0 + y1) / 2,
                (x1 - x0), (y1 - y0)]
            return np.array(b)
        
        def __getitem__(self, idx):
            path = self.paths[idx]
            y_class = self.y[idx]
            np_bb = np.fromstring(self.bb[idx][2:-1],sep=' ')
            #np_bb = self.box_xyxy_to_cxcywh(np_bb)
            x, y_bb = transformsXY(path,np_bb , self.transforms)
            x = normalize(x)
            x = np.rollaxis(x, 2)
            return x, y_class, y_bb

    train_ds = RoadDataset(X_train['new_path'],X_train['new_bb'] ,y_train,transforms=True)
    valid_ds = RoadDataset(X_val['new_path'],X_val['new_bb'],y_val)

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers=8)
    val_loader = DataLoader(valid_ds, batch_size=batch_size,num_workers=8)


    # dataset_path = "dataset/cornell"
    # img_set = "train"
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    # train_dataset = CornellDataset(dataset_path, "train", normalize)
    # val_dataset = CornellDataset(dataset_path, "val", normalize)

    # batch_size = 32
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model will be trained on {device}!!")

    model = DETR(num_class=2,number_of_embed=16).to(device) #19 + 1 background class for Cornell Dataset
    weight = torch.tensor([1,0.1]).to(device)
    print(weight.size())
    loss = HungarianMatcher(weight,num_class=2)
   

    #bbox parameters
    lr =1e-3
    optim_bbox = optim.Adam(model.parameters(), lr=lr)

    epochs = 200

    train_model = Trainer(model, train_loader, val_loader, device, optim_bbox, loss.loss, epochs)

    train_model.train_network(orientation_only=True)



    

