import numpy as np
import torch
import scipy

class HungarianMatcher():
    def __init__(self):
        pass
    @torch.no_grad()
    def forward(self,targets,output):
        '''
        targets dic of class and bbox
            class_size [num_class] only one bbox for img
            bbox size [batch,4]
        output dic of class and bbox
            class size [batch,num_query,num_class] num_class 
            bbox is size [batch,num_query,4] we are using only bounding bbox
        '''
        output_bbox = output['bbox']
        output_class = output['class']

        target_bbox = targets['bbox']
        target_class = targets['class']

        dist = torch.cdist(output_bbox, target_bbox.unsqueeze(1), p=1)
        
