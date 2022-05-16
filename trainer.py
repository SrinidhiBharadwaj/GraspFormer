import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from model import  DETR
from cornell_dataset import CornellDataset
import os

class Trainer():
    def __init__(self, model, train_loader, val_loader, device, optimizer_bbox, loss_bbox, 
                 optimizer_rot, loss_rot, epochs=10, lr=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.loss_bbox = loss_bbox
        self.loss_rot = loss_rot
        self.optimizer_bbox = optimizer_bbox
        self.optimizer_rot = optimizer_rot
        self.save_every = 5 #Save the model every n epochs

    def train_network(self, orientation_only=False):
        print("Beginning training!!")
        print(f"Number of training images: {len(self.train_loader)}")
        for epoch in range(self.epochs):
            running_orientation_loss = 0
            for t, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)

                class_label = y[0].long().to(self.device)
                bbox_label = y[1].float().to(self.device)
                
                bbox_pred, class_pred = self.model(x)
                print(bbox_pred.size(),class_pred.size())
                exit()
                if not orientation_only:
                    #Needs to be filled with loss for bbox matching(matching loss)
                    pass

                #Learn orientation model weights regardless
                orientation_loss = self.loss_rot(class_pred, class_label)
                self.optimizer_rot.zero_grad()
                orientation_loss.backward()
                self.optimizer_rot.step()
                running_orientation_loss += orientation_loss.item()
                #print(f"Epoch: {epoch}, Batch: {t}, Orientation_loss: {orientation_loss/len(self.train_loader)}")
            
            with torch.no_grad():
                running_loss_val = 0
                model.eval()
                for i, (x_val, y_val) in enumerate(self.val_loader):
                    x_val = x_val.to(device)
                    y_val = y_val[0].long().to(device)

                    _, class_pred_val = self.model(x_val, orientation_only)
                    loss_rot_val = self.loss_rot(class_pred_val, y_val)
                    running_loss_val += loss_rot_val.item()
                
                print(f"Epoch: {epoch}, Training loss: {orientation_loss/len(self.train_loader)}, \
                                Validation Loss: {running_loss_val/len(self.val_loader)}")


            #print(f"Epoch: {epoch}/{self.epochs} --> Orientation_loss: {running_orientation_loss/len(self.train_loader)} \
             #                         Bbox Matching loss: TBF")
            
            if epoch+1 == self.save_every:
                save_name = os.path.join('model_{}.ckpt'.format(epoch))
                torch.save({
                'epoch': epoch + 1,
                'model': self.model.state_dict(), # 'model' should be 'model_state_dict'
                'optimizer_bbox': self.optimizer_bbox.state_dict(),
                'optimizer_rot': self.optimizer_rot.state_dict(),
                'loss_bbox': self.loss_bbox.item(),
                'loss_rot': self.loss_rot.item(),
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

    model = DETR().to(device) #19 + 1 background class for Cornell Dataset

    #Rotation parameters
    loss_rot = nn.CrossEntropyLoss()
    lr_orientation = 1e-4
    optim_rot = optim.SGD(model.parameters(), lr=lr_orientation, momentum=0.9)

    #bbox parameters
    loss_bbox = None
    lr_bbox = 1e-2
    optim_bbox = optim.Adam(model.parameters(), lr=lr_bbox)

    epochs = 30

    train_model = Trainer(model, train_loader, val_loader, device, optim_bbox, loss_bbox, 
                            optim_rot, loss_rot, epochs)

    train_model.train_network(orientation_only=True)



    

