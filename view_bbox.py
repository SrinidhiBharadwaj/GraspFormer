
'''
Helper class to draw bounding boxes given their co-ordinates and orientation
'''
import numpy as np
from shapely.geometry import polygon
from matplotlib import pyplot as plt

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

        rotated_box = R @ (points-center)
        return rotated_box+center

    def draw_rotated_box(self, image, plot=False):
        
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

        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(image)

        plt.plot(pred_x[0:2],pred_y[0:2], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[1:3],pred_y[1:3], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[2:4],pred_y[2:4], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[3:5],pred_y[3:5], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
        
        plt.draw()
        plt.show()
        

if __name__ == "__main__":
    pass