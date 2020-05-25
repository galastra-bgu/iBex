from fastai.vision import *
from fastai.metrics import error_rate,accuracy,Precision,Recall, ConfusionMatrix
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

#init learn
def trainfor(num_epochs,size):
    np.random.seed(33)
    torch.random.manual_seed(33)
    data : ImageDataBunch= ImageDataBunch.from_folder("./image_data",train='.',valid_pct=0.2, ds_tfms=get_transforms(flip_vert=False), size=size, bs=bs,num_workers=0).normalize(imagenet_stats)
    #beware of num_workers, many errors are because of it
    learn.data = data
    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    try:
        min_grad_lr = learn.recorder.min_grad_lr
    except:
        min_grad_lr = 3e-5
    learn.fit_one_cycle(num_epochs, min_grad_lr,callbacks=[SaveModelCallback(learn, every='epoch', monitor='error_rate')])

if __name__ == "__main__":
    bs = 32 

    data : ImageDataBunch= ImageDataBunch.from_folder("./image_data",train='.',valid_pct=0.2, ds_tfms=get_transforms(flip_vert=False), size=50, bs=bs,num_workers=4).normalize(imagenet_stats)
    learn = cnn_learner(data, models.resnet50, metrics=[accuracy,Precision(),Recall(),FBeta(),ConfusionMatrix()] , callback_fns=[OverSamplingCallback])
    learn.path = Path("./classification_models/more_data/unfrozen")
    learn = learn.load("bestmodel-1269")
    # learn.eval()
    print('**********EVALUATING********')
    print(learn.validate())
    