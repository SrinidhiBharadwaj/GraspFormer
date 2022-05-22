from tabnanny import verbose
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from model import GraspFormer, detr_simplified, DETRModel, DETR
from cornell_dataset import CornellDataset
from hungarian import HungarianMatcher, HungarianMatcherOverLoad
import os

class Trainer():
    def __init__(self, model, train_loader, val_loader, device, optimizer_bbox, loss_bbox, scheduler, epochs=10, lr=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.loss_bbox = loss_bbox
        self.optimizer_bbox = optimizer_bbox
        self.save_every = 5 #Save the model every n epochs
        self.scheduler = scheduler

    def train_network(self, orientation_only=False):
        print("Beginning training!!")
        print(f"Number of training images: {len(self.train_loader)}")
        self.model = self.model.to(device)
        for epoch in range(self.epochs):
            running_val_loss = 0
            running_bbox_loss = 0
            for t, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)

                class_label = y[0].long().to(self.device)
                bbox_label = y[1].float().to(self.device)
                
                output = self.model(x)
                bbox_pred = output["pred_boxes"]
                class_pred = output["pred_logits"]
         
                #bbox_pred, class_pred = self.model(x)
        
                #Needs to be filled with loss for bbox matching(matching loss)
                #Learn orientation model weights regardless
                target_dic = {'bbox':bbox_label, 'class':class_label}
                output_dic = {'bbox':bbox_pred,'class':class_pred}

                loss = self.loss_bbox(target_dic,output_dic)
                self.optimizer_bbox.zero_grad()
                loss.backward()
                self.optimizer_bbox.step()
                running_bbox_loss += loss.item()
            
            with torch.no_grad():
                model.eval()
                for i, (x, y) in enumerate(self.val_loader):
                    x = x.to(self.device)

                    class_label = y[0].long().to(self.device)
                    bbox_label = y[1].float().to(self.device)
                    
                    output = self.model(x)
                    bbox_pred = output["pred_boxes"]
                    class_pred = output["pred_logits"]
                    #bbox_pred, class_pred = self.model(x)

                    target_dic = {'bbox':bbox_label, 'class':class_label}
                    output_dic = {'bbox':bbox_pred,'class':class_pred}

                    loss = self.loss_bbox(target_dic,output_dic)
                    running_val_loss += loss.item() 
            #self.scheduler.step()
            print(f"Epoch: {epoch}, Train Loss: {running_bbox_loss/len(self.train_loader)}, Val Loss: {running_val_loss/len(self.val_loader)}")


            #print(f"Epoch: {epoch}/{self.epochs} --> Orientation_loss: {running_orientation_loss/len(self.train_loader)} \
             #                         Bbox Matching loss: TBF")
            
            if (epoch+1) % self.save_every == 0:
                save_name = os.path.join('model_{}.ckpt'.format(epoch))
                torch.save({
                'epoch': epoch + 1,
                'model': self.model.state_dict(), # 'model' should be 'model_state_dict'
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

    model = detr_simplified(num_classes=20)
    model = DETRModel(num_classes=20, num_queries=16)
    #bbox parameters
    weight = torch.tensor([1,0.1]).to(device)

    loss_bbox = HungarianMatcherOverLoad(num_class=20)
    lr_bbox = 1e-5
    optim_bbox = optim.AdamW(model.parameters(), lr=lr_bbox)
    scheduler = lr_scheduler.StepLR(optim_bbox, step_size=10, gamma=0.3)


    epochs = 100

    train_model = Trainer(model, train_loader, val_loader, device, optim_bbox, loss_bbox.loss, scheduler, epochs)

    train_model.train_network(orientation_only=False)



    

