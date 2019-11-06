
from fastai.vision import *
from fastai.metrics import error_rate
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
data = ImageDataBunch.from_folder(image_path,train='.',valid_pct=0.2, ds_tfms=get_transforms(flip_vert=False), size=299, bs=bs,num_workers=0).normalize(imagenet_stats)

## Training: resnet50

learn = cnn_learner(data, models.resnet50, metrics=error_rate, callback_fns=[OverSamplingCallback])
learn.path = Path("./learners")

#learn.lr_find()
#learn.recorder.plot(suggestion=True)
#min_grad_lr = learn.recorder.min_grad_lr
min_grad_lr = 1e-3

print('*** started training frozen... ***')
learn.fit_one_cycle(12, min_grad_lr)
learn.save('stage-1-x8')
print('*** saved frozen ***')
learn.export()

#learn.recorder.plot_losses()
#learn.recorder.plot_lr()

"""See the error in your eyes:"""

#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
#interp.plot_top_losses(18, figsize=(15,11))

"""Now We unfreeze a second batch"""
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr
print('*** started unfrozen-1... ***')
learn.fit_one_cycle(4, min_grad_lr)
learn.save('stage-u-1-x8')
print('*** saved unfrozen-1 ****')

learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr
print('*** started unfrozen-2... ***')
learn.fit_one_cycle(4, min_grad_lr)
learn.save('stage-u-2-x8')
print('*** saved unfrozen-2 ****')

learn.export()
#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
#interp.plot_top_losses(16, figsize=(15,11))