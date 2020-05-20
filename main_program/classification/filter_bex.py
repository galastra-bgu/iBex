import sys , os, time,stat
import torch
from shutil import copyfile
from fastai.vision import *
import pandas as pd
from datetime import datetime
from PIL import Image
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

def label_images(imglist,batch_size,accuracy_thresh):
    accuracy_thresh_L,accuracy_thresh_R = accuracy_thresh
    # images2label = get_image_files(folder_path,recurse=True)
    learn = load_learner('main_program/classification//model/','trained1269.pkl',test=imglist)
    p = learn.get_preds(ds_type=DatasetType.Test,n_batch=batch_size)
    labels = p[0].argmax(-1)
    m = p[0][:,1]
    # print(m)
    for i in range(len(m)):
        if accuracy_thresh_L<m[i]<accuracy_thresh_R:
            labels[i] = 2 #not_sure 
    return labels,p[0][:,1]


def group(iterator,count):
    itr = iter(iterator)
    while True:
        yield [next(itr) for i in range(count)]

def getImageFolder(rootfolder_path,img_path):
    # print('***getImageFolder')
    prev_parent = img_path
    year_name = prev_parent
    while prev_parent.parent != Path(rootfolder_path):
        # print(str(prev_parent.name)+'-'+str(img_path.name))
        year_name  = prev_parent
        prev_parent = prev_parent.parent
    return str(prev_parent.name),str(year_name.name)


def dic2csv(filename_properties):
    
    def get_date_as_numbers(date):
        num_date = {}
        date = re.split(':| ',date)
        #NOTE: works only on trap cameras for now
        num_date['year'] = date[0]
        num_date['month'] = date[1]
        num_date['day'] = date[2]
        num_date['hour'] = date[3]
        num_date['minute'] = date[4]
        num_date['second'] = date[5]
        return num_date

    columns = ['Filename','RelativePath','Folder','Ibex Percentage','DateTime','Year','Month','Day','Hour','Minute','Second']
    arr = [list(range(len(columns))) for _ in range(len(filename_properties))] 

  

    for i,name in enumerate(filename_properties.keys()):
        num_date = get_date_as_numbers(filename_properties[name][2])
        arr[i][0] = name
        arr[i][1] = filename_properties[name][0]
        arr[i][2] = filename_properties[name][1]
        arr[i][3] = filename_properties[name][3]
        arr[i][4] = filename_properties[name][2]
        arr[i][5] = num_date['year']
        arr[i][6] = num_date['month']
        arr[i][7] = num_date['day']
        arr[i][8] = num_date['hour']
        arr[i][9] = num_date['minute']
        arr[i][10] = num_date['second']

    df = pd.DataFrame(arr,columns=columns)
    df.to_excel("output_filter.xlsx",sheet_name="filter_ibex")
    return df

#main: 
def main():
    start_time = datetime.now()
    filename_properties = {}
    total_ibex, total_no_ibex,total_not_sure = 0,0,0
    try:
        root_folder_path = sys.argv[1]
        if len(sys.argv) > 2:
            batch_size  = int(sys.argv[2])
        else:
            batch_size = 1 #relevent fixme later
        if len(sys.argv) > 4:
            accuracy_thresh_L = sys.argv[3]
            accuracy_thresh_R = sys.argv[4]
        else:
            accuracy_thresh_L = 0.1
            accuracy_thresh_R = 0.9
        accuracy_thresh = (accuracy_thresh_L,accuracy_thresh_R)
    except:
        print('Folder path not given')
        print('Enter `command <folder_path>` <batch_size> (Optional)')
    if os.path.isdir(root_folder_path):
        try:
            os.makedirs('../labeled/0',exist_ok=True)
            os.makedirs('../labeled/1')
            os.makedirs('../labeled/2')
            print('Directories labeled/0 and labaled/1 created successfully')
        except OSError as error:
            print('labeled directories already exist, no need to make them')

        images2label = get_image_files(root_folder_path,recurse=True)
        print('Found ',len(images2label),'image files')

        for imgs in group(images2label,batch_size):
            #FIXME: the last group that is < batch_size get turnacated
            y_pred,ibex_percentage = label_images(imgs,batch_size,accuracy_thresh)
            y_pred,ibex_percentage = y_pred.tolist(), ibex_percentage.tolist()
            for image_path,label,ib_per in zip(imgs,y_pred,ibex_percentage):
                new_path = '../labeled/'+str(label)
                old_name = new_path+'/'+image_path.name
                parentFolder,year = getImageFolder(root_folder_path,image_path)
                new_name = new_path+'/'+parentFolder+'-' +year+'-'+ str(image_path.name)
                shutil.copy2(image_path,new_path)
                shutil.move(old_name,new_name)
                #if label==1:
                print(image_path)
                #get datetime of the image
                try:
                    image_date = Image.open(image_path)._getexif()[36868]
                except:
                    image_date = time.ctime(os.stat(image_path)[stat.ST_MTIME])
                filename_properties[parentFolder+'-' +year+'-'+ str(image_path.name)] = (str(image_path),parentFolder,image_date, ib_per*100)
                #filename_properties[parentFolder+'-' + str(image_path.name)] = (str(image_path),parentFolder)
                if label==1:
                    total_ibex +=1
                if label==0:
                    total_no_ibex +=1
                if label==2:
                    total_not_sure +=1
        
        dic2csv(filename_properties)
        properties_file = open('filter_properties.txt',"w")
        properties_file.write("There were a total of {0} pictures.\n Folder stats:\n ibex: {1} ({2}%)\n no_ibex: {3} ({4}%)\n not_sure: {5} ({6}%)\n The accuracy threshold (L,R) is: ({7},{8})\n\n Time: {9}".format(
            len(images2label),
            total_ibex,100*total_ibex/len(images2label),
            total_no_ibex,100*total_no_ibex/len(images2label),
            total_not_sure,100*total_not_sure/len(images2label),
            accuracy_thresh_L,accuracy_thresh_R,
            datetime.now()-start_time))
        print('Done labeling! Time: ',datetime.now() - start_time)
    else:
        print('ERROR: directory does not exist')
    return filename_properties

    
if __name__ == '__main__':
    main()