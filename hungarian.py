from matplotlib.pyplot import axis
import numpy as np
import torch
import scipy
from torchvision.ops.boxes import _box_inter_union


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
    def __init__(self, num_class, weight):
        self.bound_loss = torch.nn.SmoothL1Loss()
        self.class_loss = torch.nn.CrossEntropyLoss(weight)
        
        self.num_class = num_class
        
    @torch.no_grad()
    def get_index(self,targets,output):

        output_bbox = output['bbox'].cpu().numpy()
        target_bbox = targets['bbox']
        
        ious = self.iou(output_bbox, target_bbox.unsqueeze(1).repeat(1, output_bbox.shape[1], 1).cpu().numpy())
        idx = np.argmax(ious, axis=1)

        return (idx)
    
    def loss(self,target_dic,output_dic, device="cpu"):
        batch_size = target_dic["bbox"].shape[0]
        idx = self.get_index(target_dic,output_dic) 

        bboxes_pred_per_batch =torch.zeros(batch_size, 4).to(device)
        classes_pred_per_batch = torch.zeros(batch_size, self.num_class).to(device)
        for i in range(len(idx)):
            bboxes_pred_per_batch[i, :] = output_dic['bbox'][i, idx[i], :]
            classes_pred_per_batch[i, :] = output_dic['class'][i, idx[i], :]

        bound_loss = self.bound_loss(bboxes_pred_per_batch,target_dic['bbox'].to(torch.float))

        class_label = target_dic['class']
        #print(class_label.shape, classes_pred_per_batch.shape)
        
        class_loss = self.class_loss(classes_pred_per_batch,class_label.to(torch.long))

        giou_loss = self.giou_loss(bboxes_pred_per_batch,target_dic['bbox'].to(torch.float))
        #print(giou_loss)
        return class_loss + 100*bound_loss, class_loss, 100*bound_loss

    def iou(self, box1,box2, eps=1e-7):
        '''
        box1: [bs, num_questies, 4]
        box2: [1, num_queriebs, 4]
        '''
        #print( np.maximum(box1[:, :, 0], box2[:, :, 0]))
        x1, y1 = np.maximum(box1[:, :, 0], box2[:, :, 0]), np.maximum(box1[:, :, 1], box2[:, :, 1])
        x2, y2 = np.maximum(box1[:, :, 2], box2[:, :, 2]), np.maximum(box1[:, : ,3], box2[: ,: ,3])
        inter_area = np.maximum(0, (x2 - x1 + 1)) * np.maximum(0, (y2 - y1 + 1))
        union_area = (box1[:, :, 2] - box1[:, :, 0] + 1) * (box1[:, :, 3] - box1[:, :, 1] + 1) + (box2[:, :, 2] - box2[:, :, 0] + 1) * (box2[:, :, 3] - box2[:, :, 1] + 1) - inter_area

        return inter_area/(union_area+eps)
    
    def iou_loss(self, box1,box2, eps=1e-7):
        '''
        box1: [bs, num_questies, 4]
        box2: [1, num_queriebs, 4]
        '''
        inter, union = _box_inter_union(box1, box2)
        iou = inter / union
        return (1 - iou).sum()

    def giou_loss(self, input_boxes, target_boxes, eps=1e-7):
        """
        Args:
            input_boxes: Tensor of shape (N, 4) or (4,).
            target_boxes: Tensor of shape (N, 4) or (4,).
            eps (float): small number to prevent division by zero
        """
        inter, union = _box_inter_union(input_boxes, target_boxes)
        iou = inter / union

        # area of the smallest enclosing box
        min_box = torch.min(input_boxes, target_boxes)
        max_box = torch.max(input_boxes, target_boxes)
        area_c = (max_box[:, 2] - min_box[:, 0]) * (max_box[:, 3] - min_box[:, 1])

        giou = iou - ((area_c.clamp(min=0) - union) / (area_c))

        loss = 1 - giou

        return loss.sum()