import sys , os
import torch
from shutil import copyfile
import torch
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
from detection.train_detect import get_model
import numpy as np

DETECTION_WEIGHTS = 'model/detected-w.pth'
ibex_dir = '../labeled/1/'
device = torch.device('cude') if torch.cuda.is_available() else torch.device('cpu')


def draw_detection(img_path_list,model):
    #is_image means that the img_path is actually an image, not a path(string)
    model.eval()
    ACCURACY_THRESHOLD = 0.4
    #prev_time = time.time()
    imgs = [Image.open(img_path) for img_path in img_path_list]
    imgs_size = [_img.size for _img in imgs]
    img2tensor = [i.to(device) for i in map(T.ToTensor(),imgs)]
    #img2tensor = list(_img.to(device) for _img in [T.ToTensor()(img)])
    imgs = [np.array(_img) for _img in imgs]

    print('okay, here we go')
    preds = model(img2tensor)
    print('hooray!')

    cmap = plt.get_cmap('tab20b')
    bbox_colors = [cmap(i) for i in np.linspace(0,1,5)]
    for i,detections in enumerate(preds):
        img = imgs[i]
        img_path = img_path_list[i]
        img_size = imgs_size[i]
        plt.figure()
        fig, ax = plt.subplots(1, figsize=(12,9))
        ax.imshow(img)

        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size[0] / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size[1] / max(img.shape))
        unpad_h = img_size[1] - pad_y
        unpad_w = img_size[0] - pad_x

        if detections['boxes'] is not None:
            unique_labels = detections['labels'].cpu().unique()

            # browse detections and draw bounding boxes
            for box,label_id,score in zip(detections['boxes'],detections['labels'],detections['scores']):
                if score < ACCURACY_THRESHOLD:
                    continue
                #[y_top,x_left,y_bottom,x_right]
                [x1,y1,x2,y2] = box
                label = ['background','female','kid','male adult','young adult','vanilla ibex'][label_id]
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                color = bbox_colors[label_id]
                bbox = patches.Rectangle((x1, y1), box_w, box_h,
                    linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(bbox)
                plt.text(x1, y1, s=label+' '+str(score.item()*100)[:4]+'%', 
                        color='white', verticalalignment='top',
                        bbox={'color': color, 'pad': 0})
            plt.axis('off')
            # save image
            # plt.savefig(img_path.replace(".jpg", "-det.jpg"),bbox_inches='tight', pad_inches=0.0)
            plt.savefig(img_path,bbox_inches='tight', pad_inches=0.0)

        #plt.show()



def main():
    model = get_model(8,0)
    model.load_state_dict(torch.load(DETECTION_WEIGHTS,map_location=device))
    print('Attempting to draw',len(list(os.listdir(ibex_dir))),'ibexes')

    #ibex_list = []
    for filename in os.listdir(ibex_dir):
        #ibex_list+=[os.path.join(ibex_dir,filename)]
        draw_detection(
            [os.path.join(ibex_dir,filename)],
            model
        )
    #draw_detection(ibex_list,model)

    #TODO: make csv of all the data



if __name__ == '__main__':
    main()