import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from fastai import *
from fastai.vision import *
from fastai.callbacks import *

import sys
sys.path.append('./ObjectDetection/')
sys.path.append('./Object-Detection-Metrics/lib/')

from helper.object_detection_helper import *
from loss.RetinaNetFocalLoss import RetinaNetFocalLoss
from models.RetinaNet import RetinaNet
from callbacks.callbacks import BBLossMetrics, BBMetrics, PascalVOCMetric


pics = Path("/content/drive/Shared drives/ibex pic project/imageData/1/")
df = pd.read_csv("/content/drive/Shared drives/ibex pic project/imageData/1/stav-pic-tag-export.csv")
size = 512
bs = 32

#pre-process the table
lbl_bbox = {}
for index,row in df.iterrows():
  if row['label'] in ['ibex','collared','ear tagged']: continue
  y_top,x_left,y_bottom,x_right,label = row['ymin'],row['xmin'],row['ymax'],row['xmax'],row['label']
  if row['image'] in lbl_bbox:
    tmp_list = lbl_bbox[row['image']]
    tmp_list[0].append([y_top,x_left,y_bottom,x_right])
    tmp_list[1].append(label)
    lbl_bbox[row['image']] = tmp_list
  else:
    lbl_bbox[row['image']] = [[ [y_top,x_left,y_bottom,x_right] ], [label] ]

get_y_func = lambda obj_img: lbl_bbox[obj_img.name]

def filter_func(img):
  if img.name not in lbl_bbox: return False
  try:
    return img.suffix=='.jpg'and  df[df['image']==img.name].values.tolist()[0][-1] not in ['collared','ear_tagged']
  except:
    return False

data = (ObjectItemList.from_folder(pics).filter_by_func(filter_func)
        #Where are the images? -> in coco
        .split_by_rand_pct()
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_func(get_y_func)
        #How to find the labels? -> use get_y_func
        .transform(get_transforms(), tfm_y=True, size=size)
        #.transform(size=size)
        #Data augmentation? -> Standard transforms with tfm_y=True
        .databunch(bs=bs, collate_fn=bb_pad_collate))
        #Finally we convert to a DataBunch and we use bb_pad_collate
data = data.normalize(imagenet_stats)
n_classes = data.train_ds.c
crit = RetinaNetFocalLoss(anchors)
encoder = create_body(models.resnet18, True, -2)
model = RetinaNet(encoder, n_classes=data.train_ds.c, n_anchors=18, sizes=[32,16,8,4], chs=32, final_bias=-4., n_conv=2)
#model.eval()

voc = PascalVOCMetric(anchors, size, [i for i in data.train_ds.y.classes[1:]])
learn = Learner(data, model, loss_func=crit, callback_fns=[ShowGraph, BBMetrics],
                metrics=[voc])

learn.split([model.encoder[6], model.c5top5])
learn.freeze_to(-2)

learn.lr_find()
learn.recorder.plot(suggestion=True)

learn.fit_one_cycle(3, 1e-3)

learn.unfreeze()
learn.fit_one_cycle(3, 1e-3)

data.train_ds.classes