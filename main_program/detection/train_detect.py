import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
from PIL import Image

import pandas as pd
from collections import defaultdict
import torchvision.transforms as T

from references.transforms import RandomHorizontalFlip, Compose
from references.engine import train_one_epoch, evaluate
import references.utils
from detect_bex import get_model

PICS_PATH = ("./data4detect/1/")
ANNOTATION_DIR = "./data4detect/annotation/"
DETECTION_WEIGHT = './detection_models/detected_adam.pth'
TEMP_WEIGHT_ROOT = './detection_models/tmp/'


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class iBexDataset(object):
  def __init__(self,pics_path,csv_dir,ts):
    self.ts = ts
    self.pics = pics_path
    self.lbl_bbox = defaultdict(list)
    csv_path = os.scandir(csv_dir)
    self.df = pd.concat([pd.read_csv(cv) for cv in csv_path])
    for index,row in self.df.iterrows():
      img = row['image']
      label = row['label']
      box = [x_left,y_bottom,x_right,y_top] = row['xmin'],row['ymin'],row['xmax'],row['ymax']

      self.lbl_bbox[img].append((box,label))
    self.imgs = [x for x in list(sorted(os.listdir(pics_path))) if x in self.lbl_bbox]
    for img,annot in self.lbl_bbox.items():
      for box,label in annot:
        if label!='ibex' and box not in [b for b,l in annot if l=='ibex']:
          self.lbl_bbox[img].append((box,'ibex'))
      
  

  def __getitem__(self,idx):
    def label2ix(l):
      if l=='female':
        return 1
      if l=='kid':
        return 2
      if 'adult' in l or 'mature' in l:
        return 3
      if 'young' in l:
        return 4
      if l=='ibex':
       return 5
      if 'collar' in l:
        return 6
      if 'tag' in l:
        return 7
      else:
        raise ValueError('label is undefined')

    im_id = idx
    idx = self.imgs[idx]
    img_path = os.path.join(self.pics,idx)
    img = Image.open(img_path).convert("RGB")
    if len(self.lbl_bbox[idx])==0:
      print('something bad with the lblbbox')
      print('id: ',idx)
      print('does image exist:', self.df.loc[self.df['image']==idx])
    boxes = torch.as_tensor([x[0] for x in self.lbl_bbox[idx]],dtype=torch.float32)
    labels_not_ibex = [lbl for lbl in self.lbl_bbox[idx] ]
    iscrowd = torch.tensor([0])  if len(labels_not_ibex)==1 else torch.ones((len(labels_not_ibex),),dtype=torch.int64)
    
    target = {'boxes': boxes,
              'labels':torch.as_tensor([label2ix(x[1]) for x in self.lbl_bbox[idx]]),
              'image_id': torch.tensor([im_id]),
              'area': (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0]), 
              'iscrowd': iscrowd}
    if self.ts is not None:
        img,target = self.ts(img,target)

    return img,target

  def __len__(self):
    return len(self.imgs)



def get_transform(train):
    ts = []

    ts.append(T.ToTensor())
    if train:
      ts.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])) #ImageNet properties
      ts.append(T.ColorJitter())
      ts.append(T.RandomErasing())
      ts.append(RandomHorizontalFlip(0.5))
    
    return Compose(ts)

def train(num_epochs,freeze=-1,weights=None):

    TEST_SIZE = 100
    # train on the GPU or on the CPU, if a GPU is not available
    

    # our dataset has two classes only - background and person
    num_classes = 8 # +1 for background
    # use our dataset and defined transformations
    dataset = iBexDataset(PICS_PATH,ANNOTATION_DIR, get_transform(train=True))
    dataset_test = iBexDataset(PICS_PATH,ANNOTATION_DIR, get_transform(train=False))

    #split the dataset to train and test
    indices = torch.randperm(len(dataset)).tolist()
    
    dataset = torch.utils.data.Subset(dataset,indices[:-TEST_SIZE])
    dataset_test = torch.utils.data.Subset(dataset_test,indices[-TEST_SIZE:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=references.utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=references.utils.collate_fn)

    # get the model using our helper function
    model = get_model(num_classes,freeze)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    lr = 3e-5

    if weights:
        model.load_state_dict(torch.load(DETECTION_WEIGHT))
        lr = lr/10
    optimizer = torch.optim.Adam(params,lr=lr,weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(data_loader)//2, epochs=num_epochs)

    # else:
        # optimizer = torch.optim.SGD(params, lr=lr,momentum=0.9, weight_decay=0.0005)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 100 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device) 

    torch.save(model.state_dict(),DETECTION_WEIGHT)
    return model

def main():
    model = train(8)
    torch.save(model.state_dict(),TEMP_WEIGHT_ROOT+'frozen1.pth')
    model = train(4,1,DETECTION_WEIGHT)
    torch.save(model.state_dict(),TEMP_WEIGHT_ROOT+'unfrozen_to_1.pth')
    train(4,3,DETECTION_WEIGHT)
    print("The training has been completed.")

if __name__ == '__main__':
    main()