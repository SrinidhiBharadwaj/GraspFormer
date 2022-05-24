from matplotlib.pyplot import axis
import numpy as np
import torch
import scipy

class HungarianMatcher():
    
    def __init__(self,weight=None,num_class=20):
        self.bound_loss = torch.nn.L1Loss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        if weight is not None:
            self.class_loss = torch.nn.CrossEntropyLoss(weight)
        self.num_class = num_class
        
    @torch.no_grad()
    def get_index(self,targets,output):
        '''
        targets dic of class and bbox
            class_size [num_class] only one bbox for img
            bbox size [batch,4]
        output dic of class and bbox
            class size [batch,num_query,num_class] num_class 
            bbox is size [batch,num_query,4] we are using only bounding bbox
        '''
        output_bbox = output['bbox']
        # output_class = output['class']
        target_bbox = targets['bbox']
        # target_class = targets['class']
        dist = torch.cdist(output_bbox, target_bbox.unsqueeze(1).to(torch.float), p=1)
        return torch.argmin(dist,axis=1).squeeze(1)
    
    def loss(self,target_dic,output_dic):
        batch_size = output_dic['bbox'].size(0)
        num_queries = output_dic['bbox'].size(1)
        idx = self.get_index(target_dic,output_dic)
        bound_loss = self.bound_loss(output_dic['bbox'][np.arange(batch_size),idx],target_dic['bbox'].to(torch.float))
        class_label = torch.ones(batch_size,num_queries).to(output_dic['bbox'].device)
        class_label[idx] = 0
        class_loss = self.class_loss(output_dic['class'].permute(0,2,1),class_label.to(torch.long))
        return class_loss + bound_loss


class HungarianMatcherOverLoad():
    def __init__(self, num_class):
        self.bound_loss = torch.nn.SmoothL1Loss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        
        self.num_class = num_class
        
    @torch.no_grad()
    def get_index(self,targets,output):

        output_bbox = output['bbox'].cpu().numpy()
        target_bbox = targets['bbox']
        
        ious = self.iou(output_bbox, target_bbox.unsqueeze(1).repeat(1, output_bbox.shape[1], 1).cpu().numpy())
        idx = np.argmax(ious, axis=1)

        return (idx)
    
    def loss(self,target_dic,output_dic):

        idx = self.get_index(target_dic,output_dic)  
        for i in (idx):
            bboxes_pred_per_batch = output_dic['bbox'][:, i, :]
            classes_pred_per_batch = output_dic['class'][:, i, :]
        bound_loss = self.bound_loss(bboxes_pred_per_batch,target_dic['bbox'].to(torch.float))
        class_label = target_dic['class']

        class_loss = self.class_loss(classes_pred_per_batch,class_label.to(torch.long))
        return 0.3*class_loss + 0.7*bound_loss

    def iou(self, box1,box2):
        '''
        box1: [bs, num_questies, 4]
        box2: [1, num_queriebs, 4]
        '''
        #print( np.maximum(box1[:, :, 0], box2[:, :, 0]))
        x1, y1 = np.maximum(box1[:, :, 0], box2[:, :, 0]), np.maximum(box1[:, :, 1], box2[:, :, 1])
        x2, y2 = np.maximum(box1[:, :, 2], box2[:, :, 2]), np.maximum(box1[:, : ,3], box2[: ,: ,3])
        inter_area = np.maximum(0, (x2 - x1 + 1)) * np.maximum(0, (y2 - y1 + 1))
        union_area = (box1[:, :, 2] - box1[:, :, 0] + 1) * (box1[:, :, 3] - box1[:, :, 1] + 1) + (box2[:, :, 2] - box2[:, :, 0] + 1) * (box2[:, :, 3] - box2[:, :, 1] + 1) - inter_area

        return inter_area/union_area