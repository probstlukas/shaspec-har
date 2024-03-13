import pandas as pd
import numpy as np
import os
import random
import pickle
import torch
from random import sample
from dataloaders.utils import Normalizer
freq1 = 0.3
freq2 = 20


# ========================================       Data loader Base class               =============================
class BASE_DATA():

    def __init__(self, args):
        """
        root_path                      : Root directory of the data set
        freq_save_path                 : The path to save genarated Spectrogram. If the file has already been generated, Load it directly
        window_save_path               : The path to save genarated Window index. If the file has already been generated, Load it directly
                                         This could save time by avoiding generate them again.

        data_name (Str)                : the name of data set
                                       --->[TODO]

        freq (int)                     :Sampling Frequency of the correponding dataset

        datanorm_type (Str)            : How to normalize the data 
                                       --->[standardization, minmax, per_sample_std, per_sample_minmax]

        load_all (Bool)                : This is for Freq representation data. Whether load all files in one time. this could save time by training, but it needs a lot RAM
        train_vali_quote (float)       : train vali split quote , default as 0.8

        windowsize                     :  the size of Sliding Window
        -------------------------------------------------------
		if training mode, The sliding step is 50% of the windowsize
        if test mode, The step is 10% of the windowsize. (It should be as one, But it results in to many window samples, it is difficult to generate the spectrogram)
        -------------------------------------------------------        
        drop_transition  (Bool)        : Whether to drop the transition parts between different activities
                                       ---> []

        """
        self.root_path              = args.root_path
        self.freq_save_path         = args.freq_save_path
        self.window_save_path       = args.window_save_path
        self.data_name              = args.data_name

        window_save_path = os.path.join(self.window_save_path,self.data_name)
        if not os.path.exists(window_save_path):
            os.mkdir(window_save_path)
        self.window_save_path       = window_save_path
    
        #assert self.data_name in []
        self.freq                   = args.sampling_freq  

        self.datanorm_type          = args.datanorm_type
        self.load_all               = args.load_all
        self.train_vali_quote       = args.train_vali_quote
        self.windowsize             = args.windowsize
        self.drop_transition        = args.drop_transition

        # ======================= Load the Data =================================
        self.data_x, self.data_y = self.load_all_the_data(self.root_path)
        # data_x : sub_id, sensor_1, sensor_2,..., sensor_n , sub
        # data_y : activity_id   index:sub_id
        # update col_names
        self.col_names = list(self.data_x.columns)[1:-1]

        # ======================= Generate the Sliding window for Train/Test =================================
        # train/test is different by the sliding step.
        self.train_slidingwindows = self.get_the_sliding_index(self.data_x.copy(), self.data_y.copy(), "train")
        if self.exp_mode not in ["SOCV","FOCV"]:
            self.test_slidingwindows  = self.get_the_sliding_index(self.data_x.copy(), self.data_y.copy(), "test")

        # ----------------------- TODO ----------------------------------------
        if self.exp_mode in ["SOCV","FOCV"]:
            self.num_of_cv = 5
            self.index_of_cv = 0
            self.step = int(len(self.train_slidingwindows)/self.num_of_cv)
            self.window_index_list = list(np.arange(len(self.train_slidingwindows)))
            random.shuffle(self.window_index_list)
            if self.datanorm_type is not None:
                self.normalized_data_x = self.normalization(self.data_x.copy())
            else:
                self.normalized_data_x = self.data_x.copy()

        elif self.exp_mode == "LOCV":
            self.num_of_cv = len(self.LOCV_keys)
            self.index_of_cv = 0

        else:
            self.num_of_cv = 1

    def update_train_val_test_keys(self):
        """
        It should be called at the begin of each iteration
        it will update:
        1. train_window_index
        2. vali_window_index
        3. test_window_index
        it will also:
        normalize the data , because each iteration uses different training data
        calculate the weights of each class
        """
        if self.exp_mode in ["Given", "LOCV"]:
            if self.exp_mode == "LOCV":
                print("Leave one Out Experiment : The {} Part as the test".format(self.index_of_cv+1))

                self.test_keys =  self.LOCV_keys[self.index_of_cv]
                self.train_keys = [key for key in self.all_keys if key not in self.test_keys]
                # update the index_of_cv for the next iteration
                self.index_of_cv = self.index_of_cv + 1

            # Normalization the data
            if self.datanorm_type is not None:
                train_vali_x = pd.DataFrame()
                for sub in self.train_keys:
                    temp = self.data_x[self.data_x[self.split_tag]==sub]
                    train_vali_x = pd.concat([train_vali_x, temp])

                test_x = pd.DataFrame()
                for sub in self.test_keys:
                    temp = self.data_x[self.data_x[self.split_tag]==sub]
                    test_x = pd.concat([test_x,temp])
            
                train_vali_x, test_x = self.normalization(train_vali_x, test_x)

                self.normalized_data_x = pd.concat([train_vali_x,test_x])
                self.normalized_data_x.sort_index(inplace=True)
            else:
                self.normalized_data_x = self.data_x.copy()

            # Filter the first element of the window by the keys of the test
            all_test_keys = []
            if self.split_tag == "sub":
                for sub in self.test_keys:
                    all_test_keys.extend(self.sub_ids_of_each_sub[sub])
            else:
                all_test_keys = self.test_keys.copy()

            # -----------------test_window_index---------------------
            test_file_name = os.path.join(self.window_save_path,
                                          "{}_droptrans_{}_windowsize_{}_{}_test_ID_{}.pickle".format(self.data_name, 
                                                                                                      self.drop_transition,
                                                                                                      self.exp_mode,
                                                                                                      self.windowsize, 
                                                                                                      self.index_of_cv-1))
            if os.path.exists(test_file_name):
                with open(test_file_name, 'rb') as handle:
                    self.test_window_index = pickle.load(handle)
            else:
                self.test_window_index = []
                for index, window in enumerate(self.test_slidingwindows):
                    sub_id = window[0]
                    if sub_id in all_test_keys:
                        self.test_window_index.append(index)
                with open(test_file_name, 'wb') as handle:
                    pickle.dump(self.test_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # -----------------train_vali_window_index---------------------

            train_file_name = os.path.join(self.window_save_path,
                                           "{}_droptrans_{}_windowsize_{}_{}_train_ID_{}.pickle".format(self.data_name, 
                                                                                                        self.drop_transition,
                                                                                                        self.exp_mode,
                                                                                                        self.windowsize, 
                                                                                                        self.index_of_cv-1))
            if os.path.exists(train_file_name):
                with open(train_file_name, 'rb') as handle:
                    train_vali_window_index = pickle.load(handle)
            else:
                train_vali_window_index = []
                for index, window in enumerate(self.train_slidingwindows):
                    sub_id = window[0]
                    if sub_id not in all_test_keys:
                        train_vali_window_index.append(index)
                with open(train_file_name, 'wb') as handle:
                    pickle.dump(train_vali_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

            random.shuffle(train_vali_window_index)
            self.train_window_index = train_vali_window_index[:int(self.train_vali_quote*len(train_vali_window_index))]
            self.vali_window_index = train_vali_window_index[int(self.train_vali_quote*len(train_vali_window_index)):]

        elif self.exp_mode in ["SOCV", "FOCV"]:
            print("Overlapping random Experiment : The {} Part as the test".format(self.index_of_cv+1))
            start = self.index_of_cv * self.step
            if self.index_of_cv < self.num_of_cv-1:
                end = (self.index_of_cv+1) * self.step
            else:
                end = len(self.train_slidingwindows)

            train_vali_index = self.window_index_list[0:start] + self.window_index_list[end:len(self.window_index_list)]
            self.test_window_index = self.window_index_list[start:end] 
            # copy shuffle
            self.train_window_index = train_vali_index[:int(self.train_vali_quote*len(train_vali_index))]
            self.vali_window_index = train_vali_index[int(self.train_vali_quote*len(train_vali_index)):]

            self.index_of_cv = self.index_of_cv + 1


        else:
            raise NotImplementedError

        self.act_weights = self.update_classes_weight()


    def update_classes_weight(self):
        class_transform = {x: i for i, x in enumerate(self.no_drop_activites)}

        y_of_all_windows  = []
        # get all labels of all windows
        for index in self.train_window_index:

            start_index = self.train_slidingwindows[index][1]
            end_index = self.train_slidingwindows[index][2]

            y_of_all_windows.append(class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]])
        y_of_all_windows = np.array(y_of_all_windows)


        target_count = np.array([np.sum(y_of_all_windows == label) for label in set(y_of_all_windows)])
        weight_target = 1.0 / target_count
        weight_samples = np.array([weight_target[t] for t in y_of_all_windows])
        weight_samples = torch.from_numpy(weight_samples)
        weight_samples = weight_samples.double()


        print("[-] Target sampling weights: ", weight_target)


        return weight_samples

    def load_all_the_data(self, root_path):
        raise NotImplementedError


    def normalization(self, train_vali, test=None):
        train_vali_sensors = train_vali.iloc[:,1:-1]
        self.normalizer = Normalizer(self.datanorm_type)
        self.normalizer.fit(train_vali_sensors)
        train_vali_sensors = self.normalizer.normalize(train_vali_sensors)
        train_vali_sensors = pd.concat([train_vali.iloc[:,0],train_vali_sensors,train_vali.iloc[:,-1]], axis=1)
        if test is None:
            return train_vali_sensors
        else:
            test_sensors  = test.iloc[:,1:-1]
            test_sensors  = self.normalizer.normalize(test_sensors)
            test_sensors  =  pd.concat([test.iloc[:,0],test_sensors,test.iloc[:,-1]], axis=1)
            return train_vali_sensors, test_sensors

    def get_the_sliding_index(self, data_x, data_y , flag = "train"):
        """
        Because of the large amount of data, it is not necessary to store all the contents of the slidingwindow, 
        but only to access the index of the slidingwindow
        Each window consists of three parts: sub_ID , start_index , end_index
        The sub_ID is used for train test split, if the subject train test split is applied
        """
        if os.path.exists(os.path.join(self.window_save_path,
                                       "{}_{}_drop_trans_{}_windowsize{}.pickle".format(self.data_name, 
                                                                                        flag, 
                                                                                        self.drop_transition,
                                                                                        self.windowsize))):
            print("-----------------------Sliding file are generated -----------------------")
            with open(os.path.join(self.window_save_path,
                                   "{}_{}_drop_trans_{}_windowsize{}.pickle".format(self.data_name, 
                                                                                    flag, 
                                                                                    self.drop_transition,
                                                                                    self.windowsize)), 'rb') as handle:
                window_index = pickle.load(handle)
        else:
            print("----------------------- Get the Sliding Window -------------------")

            data_y = data_y.reset_index()
            data_x["activity_id"] = data_y["activity_id"]

            if self.drop_transition:
                data_x['act_block'] = ((data_x['activity_id'].shift(1) != data_x['activity_id']) | (data_x['sub_id'].shift(1) != data_x['sub_id'])).astype(int).cumsum()
            else:
                data_x['act_block'] = (data_x['sub_id'].shift(1) != data_x['sub_id']).astype(int).cumsum()

            freq         = self.freq   
            windowsize   = self.windowsize

            if flag == "train":
                displacement = int(0.5 * self.windowsize)
                #drop_for_augmentation = int(0.2 * self.windowsize)
            elif flag == "test":
                displacement = int(0.1 * self.windowsize)
                #drop_for_augmentation = 1

            window_index = []
            for index in data_x.act_block.unique():

                temp_df = data_x[data_x["act_block"]==index]
                assert len(temp_df["sub_id"].unique()) == 1
                sub_id = temp_df["sub_id"].unique()[0]
                start = temp_df.index[0]# + drop_for_augmentation 
                end   = start+windowsize

                while end <= temp_df.index[-1]+1:# + drop_for_augmentation :

                    if temp_df.loc[start:end-1,"activity_id"].mode().loc[0] not in self.drop_activities:
                        window_index.append([sub_id, start, end])

                    start = start + displacement
                    end   = start + windowsize

            with open(os.path.join(self.window_save_path,"{}_{}_drop_trans_{}_windowsize{}.pickle".format(self.data_name, flag, self.drop_transition,windowsize)), 'wb') as handle:
                pickle.dump(window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return window_index
    

    def Sensor_filter_acoording_to_pos_and_type(self, select, filter, all_col_names, filtertype):
        """
        select  (list or None): What location should be chosen
        filter  (list or None): whether all sensors can be filtered 
        all_col_names     (list)  : Columns currently available for Filtering
        """ 
        if select is not None:
            if filter is None:
                raise Exception('This dataset cannot be selected by sensor {}!'.format(filtertype))
            else:
                col_names = []
                for col in all_col_names:
                    selected = False
                    for one_select in select:
                        assert one_select in filter
                        if one_select in col:
                            selected = True
                    if selected:
                        col_names.append(col)
                return col_names
        else:
            return None

    def regroup_and_reindex_all_cols(self, df):
        columns = df.columns[:-1]
        # big gourps
        groups = {}
        for col in columns:
            index = col.split("_")[-1]
            if index in groups.keys():
                groups[index].append(col)
            else:
                groups[index] = [col]
        # sub_groups
        index = 1
        columns_mapping = {}
        columns_groups = []
        for key in groups.keys():
            cols = groups[key]
            cols_set = []
            for col in cols:
                cols_set.append(col.split("_")[0])
            cols_set = set(cols_set)

            for col_begin in cols_set:
                sub_groups= []
                for col in cols:
                    if col.split("_")[0]==col_begin:
                        columns_mapping[col] = "_".join(col.split("_")[:-1])+"_"+str(index)
                        sub_groups.append("_".join(col.split("_")[:-1])+"_"+str(index))
                index= index+1
                if col_begin in ["acc","gyro","gravacc"]:
                    columns_groups.append(sub_groups)
        columns = ["sub_id"]+[columns_mapping[col] for col in df.columns[:-1]] + ["sub"]
        return columns,columns_groups