from torch.utils.data import Dataset
import numpy as np
import os
import pickle

from .dataloader_DSADS_har import DSADS_HAR_DATA
from .dataloader_REALDISP_har import REALDISP_HAR_DATA
data_dict = {"dsads"    : DSADS_HAR_DATA, 
             "realdisp" : REALDISP_HAR_DATA
             }

class data_set(Dataset):
    def __init__(self, args, dataset, flag):
        """
        args : a dict , In addition to the parameters for building the model, the parameters for reading the data are also in here
        dataset : It should be implmented dataset object, it contarins train_x, train_y, vali_x,vali_y,test_x,test_y
        flag : (str) "train","test","vali"
        """
        self.args = args
        self.flag = flag
        self.load_all = args.load_all
        self.data_x = dataset.normalized_data_x
        self.data_y = dataset.data_y
        if flag in ["train","vali"] or self.args.exp_mode in ["SOCV","FOCV"]:
            self.slidingwindows = dataset.train_slidingwindows
        else:
            self.slidingwindows = dataset.test_slidingwindows
        self.act_weights = dataset.act_weights

        if self.flag == "train":
            # Load train
            self.window_index =  dataset.train_window_index
            print("Train data number: ", len(self.window_index))
        elif self.flag == "vali":
            # Load vali
            self.window_index =  dataset.vali_window_index
            print("Validation data number: ",  len(self.window_index))  
        else:
            # Load test
            self.window_index = dataset.test_window_index
            print("Test data number: ", len(self.window_index))  
            
        all_labels = list(np.unique(dataset.data_y))
        to_drop = list(dataset.drop_activities)
        label = [item for item in all_labels if item not in to_drop]
        self.nb_classes = len(label)
        assert self.nb_classes == len(dataset.no_drop_activites)

        classes = dataset.no_drop_activites
        self.class_transform = {x: i for i, x in enumerate(classes)}
        self.class_back_transform = {i: x for i, x in enumerate(classes)}
        self.input_length = self.slidingwindows[0][2]-self.slidingwindows[0][1]
        self.channel_in = self.data_x.shape[1]-2

    def __getitem__(self, index):
        """
        Responsible for fetching a data sample for the model.
        """ 
        index = self.window_index[index]
        start_index = self.slidingwindows[index][1]
        end_index = self.slidingwindows[index][2]

        if self.args.sample_wise ==True:
            sample_x = np.array(self.data_x.iloc[start_index:end_index, 1:-1].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))))
        else:
            sample_x = self.data_x.iloc[start_index:end_index, 1:-1].values

        sample_y = self.class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]]

        # Add a new axis to represent channels dimension
        sample_x = np.expand_dims(sample_x, 0) 

        return sample_x, sample_y, sample_y

    def __len__(self):
        return len(self.window_index)

