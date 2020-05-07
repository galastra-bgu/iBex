import sys , os
import torch
from shutil import copyfile
import torch
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import pandas as pd
from datetime import datetime
import time


DETECTION_WEIGHTS = 'model/model0205-adam-all.pth'
#ibex_dir = '../labeled/1/'
ibex_dirs = ['../labeled/0/','../labeled/1/','../labeled/2/']
#ibex_dirs = ['../labeled/1/','../labeled/2/']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@torch.no_grad()
def draw_detection(img_path_list,model):
    model.eval()
    ACCURACY_THRESHOLD = 0.60
    #prev_time = time.time()
    imgs = [Image.open(img_path) for img_path in img_path_list]
    imgs_size = [_img.size for _img in imgs]
    img2tensor = [i.to(device) for i in map(T.ToTensor(),imgs)]
    #img2tensor = list(_img.to(device) for _img in [T.ToTensor()(img)])
    imgs = [np.array(_img) for _img in imgs]

    #print('okay, here we go')
    try:
        preds = model(img2tensor)
    except:
        preds = []
    #print('hooray!')
    labels = ['background','female','kid','male adult','young male','vanilla ibex']

    # cmap = plt.get_cmap('tab20b')
    # bbox_colors = [cmap(i) for i in np.linspace(0,1,6)]
    # for i,detections in enumerate(preds):
    #     img = imgs[i]
    #     img_path = img_path_list[i]
    #     img_size = imgs_size[i]
    #     plt.figure()
    #     fig, ax = plt.subplots(1, figsize=(12,9))
    #     ax.imshow(img)

    #     pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size[0] / max(img.shape))
    #     pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size[1] / max(img.shape))
    #     unpad_h = img_size[1] - pad_y
    #     unpad_w = img_size[0] - pad_x

    #     labels = ['background','female','kid','male adult','young male','vanilla ibex']
    #     if detections['boxes'] is not None:
    #         unique_labels = detections['labels'].cpu().unique()
            

    #         # browse detections and draw bounding boxes
    #         for box,label_id,score in zip(detections['boxes'],detections['labels'],detections['scores']):
    #             if score < ACCURACY_THRESHOLD:
    #                 continue
    #             label = labels[label_id]
    #             #[y_top,x_left,y_bottom,x_right]
    #             [x1,y1,x2,y2] = box
                
    #             box_h = ((y2 - y1) / unpad_h) * img.shape[0]
    #             box_w = ((x2 - x1) / unpad_w) * img.shape[1]
    #             y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
    #             x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
    #             color = bbox_colors[label_id]
    #             bbox = patches.Rectangle((x1, y1), box_w, box_h,
    #                 linewidth=2, edgecolor=color, facecolor='none')
    #             ax.add_patch(bbox)
    #             plt.text(x1, y1, s=label+' '+str(score.item()*100)[:4]+'%', 
    #                     color='white', verticalalignment='top',
    #                     bbox={'color': color, 'pad': 0})
    #         plt.axis('off')
    #         # save image
    #         # plt.savefig(img_path.replace(".jpg", "-det.jpg"),bbox_inches='tight', pad_inches=0.0)
    #         plt.savefig(img_path,bbox_inches='tight', pad_inches=0.0)
    #time.sleep(0.1)
    return [labels[l] for x in preds for (l,c) in zip (x['labels'],x['scores'])  if c>ACCURACY_THRESHOLD ]


def get_model(num_classes,freeze=-1):
  WEIGHTS_PATH = 'main_program/classification/model/bestmodel-1269.pth'
  pretrained_model = torch.load('main_program/classification/model/trained.pkl')['model']

  pretrained_params = torch.load(WEIGHTS_PATH)['model'] #from the classification task i.e the first step of the project
  pretrained_model.load_state_dict(pretrained_params)

  modules = list(pretrained_model.children())[:-1]
  backbone = nn.Sequential(*modules)
  #we freeze all the layers untill the last one
  if freeze < 0:
    for param in backbone.parameters():
      param.required_grad = False
  else:
    for child in list(backbone.children())[:-freeze]:
      for param in child.parameters():
        param.requires_grad = False


  # FasterRCNN needs to know the number of
  # output channels in a backbone. For resnet-50, it's 2048
  # so we need to add it here

  backbone.out_channels = 2048 

  # let's make the RPN generate 5 x 3 anchors per spatial
  # location, with 5 different sizes and 3 different aspect
  # ratios. We have a Tuple[Tuple[int]] because each feature
  # map could potentially have different sizes and
  # aspect ratios
  anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

  # let's define what are the feature maps that we will
  # use to perform the region of interest cropping, as well as
  # the size of the crop after rescaling.
  # if your backbone returns a Tensor, featmap_names is expected to
  # be [0]. More generally, the backbone should return an
  # OrderedDict[Tensor], and in featmap_names you can choose which
  # feature maps to use.
  roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                  output_size=7,
                                                  sampling_ratio=2)

  # put the pieces together inside a FasterRCNN model
  model = FasterRCNN(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
  return model

def dic2csv(preds,filename_properties):

  #assert len(preds)== len(filename_properties)
  
  columns = ['Filename','RelativePath','Folder','DateTime','Ibex Percentage(From Filter)','Total ibex','Females','Kids','Mature Males','Young Males','Undentified']
  arr = [list(range(len(columns))) for _ in range(len(preds))] 
  for i,(name,l) in enumerate(preds.items()):
      arr[i][0] = name
      arr[i][1] = filename_properties[name][0]
      arr[i][2] = filename_properties[name][1]
      arr[i][3] = filename_properties[name][2]
      arr[i][4] = filename_properties[name][3]
      arr[i][5] = len(l)
      arr[i][6] = len([x for x in l if x=='female'])
      arr[i][7] = len([x for x in l if x=='kid'])
      arr[i][8] = len([x for x in l if x=='male adult'])
      arr[i][9] = len([x for x in l if x=='young male'])
      arr[i][10] = len([x for x in l if x=='vanilla ibex'])
  
  df = pd.DataFrame(arr,columns=columns)
  return df

def main(filename_properties):
    start_time = datetime.now()
    writer = pd.ExcelWriter('output_detection.xlsx',engine = 'xlsxwriter')

    model = get_model(6)
    model.load_state_dict(torch.load(DETECTION_WEIGHTS,map_location=device))
    for dir_num,ibex_dir in enumerate(ibex_dirs):
    #dir_num = 1
    #ibex_dir = ibex_dirs[1]
      print('Attempting to draw',len(list(os.listdir(ibex_dir))),'ibexes')

    #ibex_list = []
    # for filename in os.listdir(ibex_dir):
    #     #ibex_list+=[os.path.join(ibex_dir,filename)]
    #     preds[filename]=draw_detection(
    #         [os.path.join(ibex_dir,filename)],
    #         model
    #     )
    #     time.sleep(10)
      test_list = list(os.listdir(ibex_dir))

      preds = {x: draw_detection([os.path.join(ibex_dir,x)],model) for x in test_list} 
      print('finished predicting')
      df = dic2csv(preds,filename_properties)
      #df.to_excel("output_detection.xlsx",sheet_name='detection_ibex_'+str(dir_num))
      df.to_excel(writer,sheet_name='detection_ibex_'+str(dir_num))
      df.to_excel('output_detection_'+str(dir_num)+'.xlsx',sheet_name='label'+str(dir_num)) #help if crash
      #del df
      del preds
      time.sleep(30)
    writer.save()
    print('Finished the excel part. Time: ',datetime.now() - start_time)




if __name__ == '__main__':
    main()