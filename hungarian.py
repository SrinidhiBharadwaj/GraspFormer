import numpy as np
import torch
import scipy
from torchvision.ops.boxes import box_area
from collections import Counter


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/
        The boxes should be in [x0, y0, x1, y1] format
        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all(),boxes1
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        iou, union = box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou,union

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
        output_class = output['class']
        target_bbox = targets['bbox']
        target_class = targets['class']
        prob = output_class.softmax(-1)
        bs = prob.size(0)
        dist = torch.cdist(output_bbox, target_bbox.unsqueeze(1).to(torch.float), p=2).squeeze(-1)
        
        gen_dist = torch.stack([generalized_box_iou(box_cxcywh_to_xyxy(output_bbox[num]),box_cxcywh_to_xyxy(target_bbox[num].repeat(1,1))) for num in range(bs)]).squeeze(-1)
        cost =  5*dist- 2*gen_dist#- prob[:,:,0] #- 2*gen_dist
        return torch.argmin(cost,axis=1)
    
    

    def loss(self,target_dic,output_dic,verbose=False):
        batch_size = output_dic['bbox'].size(0)
        num_queries = output_dic['bbox'].size(1)
        idx = self.get_index(target_dic,output_dic)
        if verbose:
            print(Counter(idx.cpu().detach().numpy()))
            #print(idx)
        bound_loss = self.bound_loss(output_dic['bbox'][np.arange(batch_size),idx],target_dic['bbox'].to(torch.float))
        class_label = torch.ones(batch_size,num_queries).to(output_dic['bbox'].device)#*4
       
        class_label[torch.arange(batch_size),idx] = 0#target_dic['class'].to(torch.float)
        class_loss = self.class_loss(output_dic['class'].permute(0,2,1),class_label.to(torch.long))
        gen_loss = torch.stack([generalized_box_iou(output_dic['bbox'][num,idx[num]].unsqueeze(0),target_dic['bbox'][num].unsqueeze(0)) for num in range(batch_size)])

        return class_loss + bound_loss+ gen_loss.mean()
        
