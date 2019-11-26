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

#is_ibex = lambda x :  'ibex' if x==1 else 'no-ibex'
root_folder_path = sys.argv[1]

#iterator for files
def flat_folder_iterator(root_folder):
    for d, dirs, files in os.walk(root_folder):
        for f in files:
            yield (os.path.join(d,f))

def label_images(folder_path):
    images2label = get_image_files(folder_path,recurse=True)
    learn = load_learner('./model/','trained.pkl',test=images2label)
    pred_tensor = torch.argmax(learn.get_preds(ds_type=DatasetType.Test)[0],dim=1)
    return pred_tensor

try:
    os.makedirs('./labeled/0',exist_ok=True)
    os.makedirs('./labeled/1')
    print('Directories labeled/0 and labaled/1 create successfully')
except OSError as error:
    print('labeled directories already exist, no need to make them')

pred_tensor = label_images(root_folder_path)
for image_path,label in zip(flat_folder_iterator(root_folder_path),pred_tensor.tolist()):
    shutil.copy2(image_path,'./labeled/'+str(label))
print('done labeling')