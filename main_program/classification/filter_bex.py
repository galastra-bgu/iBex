import sys , os
import torch
from shutil import copyfile
from fastai.vision import *

#loading this in order to get no errors while loading learner

class OverSamplingCallback(LearnerCallback):
    def __init__(self,learn:Learner,weights:torch.Tensor=None):
        super().__init__(learn)
        self.weights = weights

    def on_train_begin(self, **kwargs):
        ds,dl = self.data.train_ds,self.data.train_dl
        self.labels = ds.y.items
        assert np.issubdtype(self.labels.dtype, np.integer), "Can only oversample integer values"
        _,self.label_counts = np.unique(self.labels,return_counts=True)
        if self.weights is None: self.weights = torch.DoubleTensor((1/self.label_counts)[self.labels])
        self.total_len_oversample = int(self.data.c*np.max(self.label_counts))
        sampler = WeightedRandomSampler(self.weights, self.total_len_oversample)
        self.data.train_dl = dl.new(shuffle=False, sampler=sampler)

#iterator for files
def flat_folder_iterator(root_folder):
    for d, dirs, files in os.walk(root_folder):
        for f in files:
            yield (os.path.join(d,f))

def label_images(folder_path):
    images2label = get_image_files(folder_path,recurse=True)
    learn = load_learner('main_program/classification//model/','trained1269.pkl',test=images2label)
    p = learn.get_preds(ds_type=DatasetType.Test,n_batch=8)
    return p[0].argmax(dim=-1)

#main: 
def main():
    try:
        root_folder_path = sys.argv[1]
    except:
        print('Folder path not given')
        print('Enter `command <folder_path>`')
    if os.path.isdir(root_folder_path):
        try:
            os.makedirs('../labeled/0',exist_ok=True)
            os.makedirs('../labeled/1')
            print('Directories labeled/0 and labaled/1 created successfully')
        except OSError as error:
            print('labeled directories already exist, no need to make them')

        pred_tensor = label_images(root_folder_path)
        for image_path,label in zip(flat_folder_iterator(root_folder_path),pred_tensor.tolist()):
            shutil.copy2(image_path,'../labeled/'+str(label))
        print('Done labeling!')
    else:
        print('ERROR: directory does not exist')
if __name__ == '__main__':
    main()