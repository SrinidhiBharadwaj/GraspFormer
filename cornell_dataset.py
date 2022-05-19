import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from skimage import io
import cv2
import os


class CornellDataset(Dataset):
    def __init__(self, path, images_set, transform=None):
        super(CornellDataset, self).__init__()
        self.dataset_path = path
        self.imgset = images_set
        self.transform = transforms.Compose([transforms.ToTensor(), transform]) 
        self.classes = ('__background__',
                         'bin_01', 'bin_02', 'bin_03', 'bin_04', 'bin_05',
                         'bin_06', 'bin_07', 'bin_08', 'bin_09', 'bin_10',
                         'bin_11', 'bin_12', 'bin_13', 'bin_14', 'bin_15',
                         'bin_16', 'bin_17', 'bin_18', 'bin_19')
        self.empty_list = []
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self.height = 224
        self.width = 224
        rect_dataset = self.fetch_gt_rect()
        self.rect_dataset_bbox = rect_dataset[0]
    

    def __len__(self):
        return len(self.rect_dataset_bbox) 

    def __getitem__(self, index):

        #Note: Index being passes is different than the image name, hence this weird synchronoization
        image_path = self.get_image_path(self.img_idx[index])
        img = io.imread(image_path)
        img = self.transform(img)
        #print(img.size())
        gt_img_rect = self.rect_dataset_bbox[index]
        gt_img_class = gt_img_rect[0]
        gt_img_box = gt_img_rect[1]
        
        gt_img_class = np.sort(gt_img_class)
        #If the image has only one class
        #print(gt_img_class.size)
        if (gt_img_class.size == 1):
            gt_class = gt_img_class[0]
            gt_bbox = gt_img_box[0]

        else:
            classes, count = np.unique(gt_img_class, return_counts=True)
            if (classes[0] == 0) and (np.around(count[0]/count.sum())>np.around(1./count.size)):
                gt_class = gt_img_class[0]
                gt_bbox = gt_img_box[0]
            else:
                class_idx = np.argmax(count)
                gt_class = classes[class_idx]
                bbox_idx = np.where(gt_img_class == gt_class)[0]
                gt_bbox = gt_img_box[bbox_idx[0]]
        
        #Convert to tensors
        gt_class_bbox = {}
        gt_class_bbox['labels'] = torch.as_tensor(gt_class, dtype=torch.long)
        gt_class_bbox['boxes'] = torch.as_tensor(gt_bbox, dtype=torch.float32)
  
        return img, gt_class_bbox  

    #Helper functions
    def get_image_path(self, index):
        filetype = ".png"
        path = os.path.join(self.dataset_path, "Images", index+filetype)
        return path

    def get_img_idx(self):
        file = os.path.join(self.dataset_path, "ImageSets", self.imgset+".txt")
        with open(file) as f:
            #Strip to remove the \n at the end of the line
            image_idx = [i.strip() for i in f.readlines()]
        return image_idx

    def get_coordinates(self, index):
        file = os.path.join(self.dataset_path, "Annotations", index+".txt")
        #print(index, file)
        #print(f"Reading from {file}")

        if (os.stat(file).st_size) == 0:
            self.empty_list.append(index)
            #print('Empty files: {}'.format(file))
        else:
            with open(file) as f:
                file_content = f.readlines()
            
            #Storing bounding box coordinartes and class information
            gt_rect = np.zeros((len(file_content), 4), dtype=np.uint8)
            gt_class = np.zeros((len(file_content)), dtype=np.int32)

            #Loop through the data (line) and store
            i = 0
            for line in file_content:
                #Data is of format [class, x1, y1, x2, y2]
                line_content = line.split()
                #print(line_content)
                
                gt_class[i] = int(line_content[0])
                x1 = float(line_content[1])
                x2 = float(line_content[3])
                y1 = float(line_content[2])
                y2 = float(line_content[4])
                if ((x1 < 0) or (x1 > self.width) or (x2 < 0) or (x2 > self.width) or (y1 < 0) or (y1 > self.height) or (y2 < 0) or (y2 > self.height)):
                    continue
                gt_rect[i, : ] = [x1, y1, x2, y2]
                i+=1
            return gt_class, gt_rect

    def fetch_gt_rect(self):
        self.img_idx = self.get_img_idx()
        gt_rect = []
        data_len = len(self.img_idx)
        # for idx in range(data_len):
        #     #Each index has a tuple gt_rect[idx][0] = class and [1] = box coordinates
        #     gt_rect.append(self.get_coordinates(self.img_idx[idx]))
        gt_rect = [self.get_coordinates(image) for image in self.img_idx]

        for idx in self.empty_list:
            self.img_idx.remove(idx)
        for i in range(gt_rect.count(None)):
            gt_rect.remove(None)

        gt_rect_dataset = [gt_rect, self.img_idx]
        return gt_rect_dataset

#Test code for the class
if __name__ == "__main__":
    dataset_path = "dataset/cornell"
    img_set = "train"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    #Image to tensor conversion is made implicit inside the class
    dataset = CornellDataset(dataset_path, img_set, transform=normalize)
    print(len(dataset))
    img, gt_class_bbox = dataset.__getitem__(321)
    bbox = (gt_class_bbox[1]).numpy()
    print(f"bbox: {bbox}")
    
    # img = np.transpose(img,(1,2,0)).numpy().astype(np.uint8).copy() 
    # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    # cv2.imwrite("DatasetCheck.png", img)
    
# import torch
# from torch.utils.data import Dataset
# import torchvision
# from torchvision import transforms
# #import imageio
# from skimage import io
# import os
# import numpy as np
# import statistics
# import matplotlib.pyplot as plt
# import pickle

# class CornellDataset(Dataset):
#     def __init__(self, path, images_set, transform=None):
#         """transforms.ToTensor(): Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
#         Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
#         [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
#         """
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         self.transform = transforms.Compose([transforms.ToTensor(), normalize]) 
#         #self.name = name
#         self.image_set = images_set
#         self.dataset_path = path
#         self.classes = ('__background__', # always index 0
#                          'bin_01', 'bin_02', 'bin_03', 'bin_04', 'bin_05',
#                          'bin_06', 'bin_07', 'bin_08', 'bin_09', 'bin_10',
#                          'bin_11', 'bin_12', 'bin_13', 'bin_14', 'bin_15',
#                          'bin_16', 'bin_17', 'bin_18', 'bin_19')
#         self.num_classes = len(self.classes)
#         self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        
#         self.image_ext = ['.png']
#         self.img_width = 224
#         self.img_height = 224
#         self.txt_empty_list = [] # updated in func load_annotation(.)
#         gt_rectdb = self.get_rectdb()
#         self.gt_rectdb = gt_rectdb[0]
#         self.image_indices = gt_rectdb[1]
        
#     def __getitem__(self, index): # index: int i.e. 0, 1, 2
#         img = io.imread(self.image_path_at(index))
#         print(self.image_path_at(index))
#         #print('img.shape: {0}'.format(img.shape))
#         gt_rects_org = self.gt_rectdb[index]
        
#         gt_classes = gt_rects_org['gt_classes']
#         gt_rects = gt_rects_org['gt_rects']
        
#         gt_classes = np.sort(gt_classes)
#         #print('sorted gt_classes: {}'.format(gt_classes))
#         # print('gt_classes.size: {}'.format(gt_classes.size))
#         if (gt_classes.size == 1):
#             gt_cls = gt_classes[0]
#             gt_rect = gt_rects[0]
#         else:
#             unique, counts = np.unique(gt_classes, return_counts=True)
#             #print('unique: {}, counts: {}'.format(unique, counts))
#             # background class
#             if (unique[0] == 0) and (np.around(counts[0]/counts.sum(),decimals=2) > np.around(1./counts.size, decimals=2)):
#                 gt_cls = gt_classes[0]
#                 gt_rect = gt_rects[0]
#                 #print('gt_cls: {}, gt_rect: {}'.format(gt_cls, gt_rect))
#             else: 
#                 # Get median index                   
#                 #i = np.argsort(gt_classes)[gt_classes.size//2]
#                 # Get the cls having max frequency
#                 i = np.argmax(counts)
#                 gt_cls = unique[i]
#                 j = np.where(gt_classes == gt_cls)[0]
#                 gt_rect = gt_rects[j[0]]
#                 #print('i: {}, gt_cls: {}, gt_rect: {}'.format(i, gt_cls, gt_rect))

        
#         '''
#         num_rects = len(gt_classes)
#         i = np.random.randint(num_rects)
#         gt_cls = gt_classes[i]
#         gt_rect = gt_rects[i]   
#         '''
        
#         #print('gt_cls: {}'.format(gt_cls))
#         #print('gt_rect: {}'.format(gt_rect))
#         gt_cls = torch.tensor(gt_cls)
#         gt_rect = torch.tensor(gt_rect)
#         gt_rect = [gt_cls, gt_rect]
#         img = torch.from_numpy(img)
#         #img = self.transform(img)
        
#         return img, gt_rect
    
#     def __len__(self):
#         return len(self.gt_rectdb)
    
#     def load_img_set_ind(self):
#         '''
#         return all image indices: pcd0101r_preprocessed_1, etc.
#         '''
#         image_set_file = os.path.join(self.dataset_path, 'ImageSets', 
#                                         self.image_set + '.txt')
#         with open(image_set_file) as f:
#             image_indices = [x.strip() for x in f.readlines()]
#         return image_indices
        
#     def image_path_from_index(self, index):
#         """
#         Construct an image path from the image's "index" identifier.
#         index is pcd0101r_preprocessed_1 for example
#         """
#         for ext in self.image_ext:
#             image_path = os.path.join(self.dataset_path, 'Images', index + ext)
#             #print('image_path: {0}'.format(image_path))
#             if os.path.exists(image_path):
#                 break
#         assert os.path.exists(image_path), \
#                 'Path does not exist: {}'.format(image_path)
#         return image_path
        
#     def image_path_at(self, i):
#         """
#         Return the absolute path to image i in the image sequence.
#         """
#         return self.image_path_from_index(self.image_indices[i])
    
#     def load_annotation(self, index):
#         '''
#         Load cls, and rect in an image
#         index is pcd0101r_preprocessed_1 for example
#         '''
        
#         filename = os.path.join(self.dataset_path, 'Annotations', index + '.txt')
        
#         # if the file is empty
#         if (os.stat(filename).st_size) == 0:
#             self.txt_empty_list.append(index)
#             print('Empty files: {}'.format(filename))
#         else:
#             #print('Loading: {}'.format(filename))
#             with open(filename) as f:
#                 data = f.readlines()
        
#             num_objs = len(data)
#             #print('Num of rects in image {0} is {1}'.format(index, len(data)))            
            
#             gt_rects = np.zeros((num_objs, 4), dtype=np.uint8)
#             gt_classes = np.zeros((num_objs), dtype=np.int32)
            
#             # Load object rects into a data frame.
#             for i, line in enumerate(data):
#                 # strip(): deletes white spaces from the begin and the end of line
#                 # split(): splits line into elements of a list by space separator
#                 obj = line.strip().split()
#                 if len(obj) != 5: # cls x1 y1 x2 y2
#                     continue
                
#                 cls = int(obj[0])
#                 x1 = float(obj[1]) 
#                 y1 = float(obj[2]) 
#                 x2 = float(obj[3]) 
#                 y2 = float(obj[4])
                
#                 if ((x1 < 0) or (x1 > self.img_width) or (x2 < 0) or (x2 > self.img_width) or (y1 < 0) or (y1 > self.img_height) or (y2 < 0) or (y2 > self.img_height)):
#                     continue
                
#                 gt_classes[i] = cls
#                 gt_rects[i, :] = [x1, y1, x2, y2]
            
#             return {'gt_classes': gt_classes, 'gt_rects': gt_rects}
    
#     def get_rectdb(self):
#         """
#         Return the database of ground-truth rects.
#         This function loads/saves from/to a cache file to speed up future calls.
#         """
       
#         self.image_indices = self.load_img_set_ind()
#         gt_rectdb = [self.load_annotation(index)
#                     for index in self.image_indices]
                        
#         # remove elements that have empty txt file
#         for idx in self.txt_empty_list:
#             self.image_indices.remove(idx)
#         for i in range(gt_rectdb.count(None)):
#             gt_rectdb.remove(None)
        
        
#         gt_rectdb = [gt_rectdb, self.image_indices]
#         # with open(cache_file, 'wb') as fid:
#         #     pickle.dump(gt_rectdb, fid, pickle.HIGHEST_PROTOCOL)
#         # print('writing gt rectdb to {}'.format(cache_file))

#         return gt_rectdb

# if __name__ == '__main__':
#     name = 'grasp'
#     dataset_path = 'dataset/cornell'
#     image_set = 'train'
#     inv_normalize = transforms.Normalize(
#                             mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#                             std=[1/0.229, 1/0.224, 1/0.225])
    
#     train_dataset = CornellDataset(name, image_set, dataset_path)
#     print('len(train_dataset): {0}'.format(len(train_dataset)))
#     #print('image_indices: {0}'.format(train_dataset.image_indices))
#     #print('txt_empty_list: {0}'.format(train_dataset.txt_empty_list))
#     print('len(gt_rectdb): {0}'.format(len(train_dataset.gt_rectdb)))
#     #print('gt_rectdb: {0}'.format(train_dataset.gt_rectdb))
        

#     img, gt_rect = train_dataset.__getitem__(11)
#     img = inv_normalize(img)
#     # CxHxW -> HxWxC
#     img = np.transpose(img,(1,2,0))
#     print('gt_cls: {0},\n gt_rect: {1}'.format(gt_rect[0], gt_rect[1]))
#     plt.imshow(img)
#     plt.show()

