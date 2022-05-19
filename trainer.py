from telnetlib import DET
import numpy as np
from numpy import ones
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from model import GraspFormer, detr_simplified, DETRModel
from cornell_dataset import CornellDataset
#from hungarian import HungarianMatcher
import os
import sys
sys.path.append('./detr/')

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

class Trainer():
    def __init__(self, model, train_loader, val_loader, device, optimizer, criterion, 
                 epochs=10, lr=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.criterion = criterion
       
        self.optimizer = optimizer
        self.save_every = 5 #Save the model every n epochs

    def train_network(self):
        print("Beginning training!!")
        print(f"Number of batches: {len(self.train_loader)}")

        model.train()
        self.criterion.train()
        for epoch in range(self.epochs):
            running_bbox_loss = 0
            for t, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                # class_label = y[0].float().to(self.device)
                # bbox_label = y[1].float().to(self.device)
                y = [{k: v.to(device) for k, v in y.items()}]
                output = self.model(x)

                loss_dict = self.criterion(output, y)
                weight_dict = self.criterion.weight_dict
        
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
       
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                running_bbox_loss += losses.item()
                #print(f"Epoch: {epoch}, Batch:{t}, Loss:{losses.item()}")            
           
                    
            print(f"Epoch:{epoch}, Training Loss: {running_bbox_loss/len(self.train_loader)}")
            
            if (epoch % self.save_every) == 0:
                save_name = os.path.join('model_{}.ckpt'.format(epoch))
                torch.save({
                'epoch': epoch + 1,
                'model': self.model.state_dict(), # 'model' should be 'model_state_dict'
                'optimizer_bbox': self.optimizer.state_dict(),
                'loss': np.mean(running_bbox_loss),
                }, save_name) 
        
        print("Finished training!")

if __name__ == "__main__":

    dataset_path = "dataset/cornell"
    img_set = "train"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    train_dataset = CornellDataset(dataset_path, "train", normalize)
    val_dataset = CornellDataset(dataset_path, "val", normalize)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model will be trained on {device}!!")

    num_classes = 20
    model = detr_simplified(num_classes).to(device) #19 + 1 background class for Cornell Dataset

    #Rotation parameters
    loss_rot = nn.CrossEntropyLoss()
    lr_orientation = 1e-4
    optim_rot = optim.SGD(model.parameters(), lr=lr_orientation, momentum=0.9)

    #bbox parameters
    matcher = HungarianMatcher()
    weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes-1, matcher, weight_dict, eos_coef = 0.5, losses=losses)


    lr = 1e-1
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 30
    model = DETRModel(num_classes, num_queries=100).to(device)
    train_model = Trainer(model, train_loader, val_loader, device, optimizer, criterion, epochs)
    train_model.train_network()



    

