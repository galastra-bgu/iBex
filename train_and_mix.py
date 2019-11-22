
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

image_path = Path("./image_data/")
bs = 64 
np.random.seed(33)
data = ImageDataBunch.from_folder(image_path,train='.',valid_pct=0.2, ds_tfms=get_transforms(flip_vert=False), size=512, bs=bs,num_workers=0).normalize(imagenet_stats)

## Training: resnet50

learn = cnn_learner(data, models.resnet50, metrics=error_rate , callback_fns=[OverSamplingCallback]).mixup()
learn.load('best-x8')
learn.path = Path("./learners/mixup")
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
min_grad_lr = 1e-4
try:
    min_grad_lr = learn.recorder.min_grad_lr
except:
    min_grad_lr = 1e-4
print('*** started training frozen... ***')
learn.fit_one_cycle(12, min_grad_lr,callbacks=[SaveModelCallback(learn, every='epoch', monitor='error_rate')])
learn.save('mixup-x8-12')
print('*** saved frozen ***')
learn.export()