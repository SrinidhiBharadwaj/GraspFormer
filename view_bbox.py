
'''
Helper class to draw bounding boxes given their co-ordinates and orientation
'''
import numpy as np
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from cornell_dataset import CornellDataset
from torchvision import transforms
import cv2

class draw_bbox():
    def __init__(self, rect, orientation):
        self.rect = rect
        self.orientation = orientation

    def calc_orientation_angle(self, ori_score):
        return 

    def rotate_box(self, points, center, angle=0):
        '''
        Rotate the given set of points using 2D rotation matrix
        R = [[Cos(angle), sin(angle)],
             [-sin(angle), cos(angle)]]
        '''
        R = np.array([[np.cos(angle), np.sin(angle)],
                      [-np.sin(angle), np.cos(angle)]])
        
        rotated_box = R @ (points-center).T
        return rotated_box.T+center

    def draw_rotated_box(self, image, plot=False, fig_num=0):
        
        points = np.array([[self.rect[0],self.rect[1]], 
                            [self.rect[2], self.rect[1]], 
                            [self.rect[2], self.rect[3]], 
                            [self.rect[0], self.rect[3]]])
        center = np.array([(self.rect[0] + self.rect[2])/2, (self.rect[1] + self.rect[3])/2])
        #Angle to radians
        orientation = -np.pi/2-np.pi/20*(self.orientation - 1)
        rotated_bbox = self.rotate_box(points, center, orientation)

        #Ref: https://stackoverflow.com/questions/30457089/how-to-create-a-shapely-polygon-from-a-list-of-shapely-points
        poly = Polygon([(rotated_bbox[0,0],rotated_bbox[0,1]), 
                                        (rotated_bbox[1,0], rotated_bbox[1,1]), 
                                        (rotated_bbox[2,0], rotated_bbox[2,1]), 
                                        (rotated_bbox[3,0], rotated_bbox[3,1])])
        pred_x, pred_y = poly.exterior.xy
        # print(pred_x, pred_y)
        fig, ax = plt.subplots(1)
        ax.imshow(image, aspect='equal')


        plt.plot(pred_x[0:2],pred_y[0:2], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[1:3],pred_y[1:3], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[2:4],pred_y[2:4], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[3:5],pred_y[3:5], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
        
        plt.draw()
        plt.savefig("Testfig_"+str(fig_num)+".png") 
        plt.show()
               

if __name__ == "__main__":
    dataset_path = "dataset/cornell"
    img_set = "train"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    inv_normalize = transforms.Normalize(
                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                            std=[1/0.229, 1/0.224, 1/0.225])
    #Image to tensor conversion is made implicit inside the class
    dataset = CornellDataset(dataset_path, img_set, transform=normalize)
    img, gt_class_bbox = dataset.__getitem__(25)
    bbox = (gt_class_bbox[1]).numpy() * 224
    bbox = bbox.astype(np.uint8)
    rot_class = gt_class_bbox[0].numpy()

    print(rot_class, bbox)
    fig, ax = plt.subplots(1)
    ax.imshow(inv_normalize(img).permute(1, 2, 0).numpy(), aspect='equal')
    bbox_draw = draw_bbox(bbox, rot_class)
    plt.savefig("Testfig.png") 
   # bbox_draw.draw_rotated_box(inv_normalize(img).permute(1, 2, 0).numpy())