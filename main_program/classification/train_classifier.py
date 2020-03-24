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

bs = 24 

size = 100
np.random.seed(33)
torch.random.manual_seed(33)
data : ImageDataBunch= ImageDataBunch.from_folder("./image_data",train='.',valid_pct=0.2, ds_tfms=get_transforms(flip_vert=False), size=size, bs=bs,num_workers=4).normalize(imagenet_stats)
#beware of num_workers, many errors are because of it

## Training: resnet50

learn = cnn_learner(data, models.resnet50, metrics=[accuracy,Precision(),Recall(),FBeta()] , callback_fns=[OverSamplingCallback])
learn.path = Path("./learners/endgame/frozen")

learn.lr_find()
learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr

print('*** started training frozen... ***')
learn.fit_one_cycle(12, min_grad_lr,callbacks=[SaveModelCallback(learn, every='epoch', monitor='error_rate')])

size = 256
np.random.seed(33)
torch.random.manual_seed(33)
data : ImageDataBunch= ImageDataBunch.from_folder("./image_data",train='.',valid_pct=0.2, ds_tfms=get_transforms(flip_vert=False), size=size, bs=bs,num_workers=4).normalize(imagenet_stats)
learn.path = Path("./learners/endgame/unfrozen")
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
try:
    min_grad_lr = learn.recorder.min_grad_lr
except:
    min_grad_lr = 3e-5
print('*** started unfrozen... ***')

learn.fit_one_cycle(4, min_grad_lr,callbacks=[SaveModelCallback(learn, every='epoch', monitor='error_rate')])

#Now we make the size larger
size = 512
np.random.seed(33)
torch.random.manual_seed(33)
data : ImageDataBunch= ImageDataBunch.from_folder("./image_data",train='.',valid_pct=0.2, ds_tfms=get_transforms(flip_vert=False), size=size, bs=bs,num_workers=4).normalize(imagenet_stats)
learn.data = data

learn.path = Path('./learners/endgame/bigger')
learn.lr_find()
learn.recorder.plot(suggestion=True)
try:
    min_grad_lr = learn.recorder.min_grad_lr
except:
    min_grad_lr = 1e-6
print('*** started unfrozen-bigger.... ***')
learn.fit_one_cycle(6, min_grad_lr,callbacks=[SaveModelCallback(learn, every='epoch', monitor='error_rate')])




learn.export()