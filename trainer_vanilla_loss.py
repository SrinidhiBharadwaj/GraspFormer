import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from model import GraspFormer, detr_simplified, DETRModel, DETR
from cornell_dataset import CornellDataset
from hungarian import HungarianMatcher, HungarianMatcherOverLoad
import os
from torch.utils.tensorboard import SummaryWriter


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
        self.writer = SummaryWriter()

    def train_network(self, orientation_only=False):
        print("Beginning training!!")
        print(f"Number of training images: {len(self.train_loader)}")
        self.model = self.model.to(device)
        for epoch in range(self.epochs):
            running_val_loss = 0
            running_train_loss = 0
            running_class_loss, running_box_loss = 0, 0
            model.train()
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
                #print(bbox_label.shape)
                target_dic = {'bbox':bbox_label, 'class':class_label}
                output_dic = {'bbox':bbox_pred,'class':class_pred}

                loss, c_loss, b_loss = self.loss_bbox(target_dic,output_dic, self.device)
                self.optimizer_bbox.zero_grad()
                loss.backward()
                self.optimizer_bbox.step()
                running_train_loss += loss.item()
                running_class_loss += c_loss.item()
                running_box_loss +=  b_loss.item()

            self.scheduler.step(running_train_loss)

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

                    loss, _, _ = self.loss_bbox(target_dic,output_dic, self.device)
                    running_val_loss += loss.item()
                    

            self.writer.add_scalar("Train Loss", running_train_loss/len(self.train_loader), epoch)
            self.writer.add_scalar("Val Loss", running_val_loss/len(self.val_loader), epoch)
            self.writer.add_scalar("Class Loss", running_class_loss/len(self.train_loader), epoch)
            self.writer.add_scalar("Box Loss", running_box_loss/len(self.train_loader), epoch)
            
            print(f"Epoch: {epoch}, Class_loss: {running_class_loss/len(self.train_loader)}, Box Loss: {running_box_loss/len(self.train_loader)}, Total Loss: {running_train_loss/len(self.train_loader)}, Val Loss: {running_val_loss/len(self.val_loader)}")
            
            if (epoch+1) % self.save_every == 0:
                save_name = os.path.join('models/model_{}.ckpt'.format(epoch+1))
                torch.save({
                'epoch': epoch+1,
                'model': self.model.state_dict(), # 'model' should be 'model_state_dict'
                }, save_name) 

        self.writer.flush()
        self.writer.close()

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
    print(len(train_loader)*batch_size)
    print(len(val_loader)*batch_size)
    p
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model will be trained on {device}!!")

    #model = detr_simplified(num_classes=20)
    model = DETRModel(num_classes=21, num_queries=2)
    #model = DETR(num_class=20)
    #bbox parameters
    
    weight = 10 * torch.ones(21).to(device)
    weight[20] = 0.1
    loss_bbox = HungarianMatcherOverLoad(num_class=21, weight=weight)
    lr_bbox = 1e-4
    optim_bbox = optim.AdamW(model.parameters(), lr=lr_bbox)
    scheduler = lr_scheduler.ReduceLROnPlateau(optim_bbox, 'min', verbose=True)
    epochs = 500

    train_model = Trainer(model, train_loader, val_loader, device, optim_bbox, loss_bbox.loss, scheduler, epochs)

    train_model.train_network(orientation_only=False)



    
