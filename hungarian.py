import numpy as np
import torch
import scipy

class HungarianMatcher():
    
    def __init__(self,weight=None,num_class=20):
        self.bound_loss = torch.nn.MSELoss()
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
        return torch.argmax(dist,axis=1).squeeze(1)
    
    def loss(self,target_dic,output_dic):
        batch_size = output_dic['bbox'].size(0)
        num_queries = output_dic['bbox'].size(1)
        idx = self.get_index(target_dic,output_dic)
        bound_loss = self.bound_loss(output_dic['bbox'][np.arange(batch_size),idx],target_dic['bbox'].to(torch.float))
        class_label = torch.ones(batch_size,num_queries).to(output_dic['bbox'].device)
        class_label[idx] = 0
        class_loss = self.class_loss(output_dic['class'].permute(0,2,1),class_label.to(torch.long))
        return class_loss + bound_loss
        
