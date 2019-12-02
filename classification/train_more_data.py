
from fastai.vision import *
from fastai.metrics import error_rate
from fastai.callbacks import *
from torch.utils.data.sampler import WeightedRandomSampler

__all__ = ['OverSamplingCallback']

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
bs = 64 
size = 299
np.random.seed(33)
#download images:
download_images('../misc/downloads/download_links','../image_data/1/',max_pics=500)

data = ImageDataBunch.from_folder("./image_data",train='.',valid_pct=0.2, ds_tfms=get_transforms(flip_vert=False), size=size, bs=bs,num_workers=0).normalize(imagenet_stats)
## Training: resnet50

learn = cnn_learner(data, models.resnet50, metrics=error_rate , callback_fns=[OverSamplingCallback])
learn.load('best-x8')
learn.path = Path("./learners/more_data/frozen")

learn.lr_find()
learn.recorder.plot(suggestion=True)
try:
    min_grad_lr = learn.recorder.min_grad_lr
except:
    min_grad_lr = 1e-4

print('*** started training frozen... ***')
learn.fit_one_cycle(4, min_grad_lr,callbacks=[SaveModelCallback(learn, every='epoch', monitor='error_rate')])
print('*** saved frozen ***')
learn.export('frozen-moredata')

"""Now We unfreeze a second batch"""
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
try:
    min_grad_lr = learn.recorder.min_grad_lr
except:
    min_grad_lr = 1e-7
print('*** started unfrozen... ***')
learn.path = Path("./learners/more_data/unfrozen")
learn.fit_one_cycle(4, min_grad_lr,callbacks=[SaveModelCallback(learn, every='epoch', monitor='error_rate')])

learn.export('unfrozen-moredata')
#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
#interp.plot_top_losses(16, figsize=(15,11))