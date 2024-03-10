import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import os
import numpy as np
import time
from dataloaders import data_dict,data_set
from sklearn.metrics import confusion_matrix
# import models
from models.model_builder import model_builder

from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from utils import EarlyStopping, adjust_learning_rate_class

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataloaders.augmentation import RandomAugment, mixup_data
import random
import os

from tqdm import tqdm
import json
from math import ceil

class MixUpLoss(nn.Module):
    """
    Mixup implementation heavily borrowed from https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py#L42
    Adapt the loss function `crit` to go with mixup.
    """

    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        self.reduction = reduction

    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output, target[:, 0].long()), self.crit(output, target[:, 1].long())
            d = loss1 * target[:, 2] + loss2 * (1 - target[:, 2])
        else:
            d = self.crit(output, target)
        if self.reduction == 'mean':
            return d.mean()
        elif self.reduction == 'sum':
            return d.sum()
        return d

    def get_old(self):
        if hasattr(self, 'old_crit'):
            return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit
        
    

class Exp(object):
    def __init__(self, args):
        self.args = args

        # set the device
        self.device = self.acquire_device()

        self.optimizer_dict = {"Adam": optim.Adam}
        self.criterion_dict = {"MSE": nn.MSELoss, "CrossEntropy": nn.CrossEntropyLoss}

        # Biuld The Model
        self.model  = self.build_model().to(self.device)
        print("Done!")
        self.args.model_size = np.sum([para.numel() for para in self.model.parameters() if para.requires_grad])
        print("Parameter:", self.args.model_size)


        print("Set the seed as: ", self.args.seed)

    def acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def build_model(self):
        model = model_builder(self.args)
        return model.double()

    def _select_optimizer(self):
        if self.args.optimizer not in self.optimizer_dict.keys():
            raise NotImplementedError
        model_optim = self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        if self.args.criterion not in self.criterion_dict.keys():
            raise NotImplementedError
        criterion = self.criterion_dict[self.args.criterion]()
        return criterion

    # 
    def _get_data(self, data, flag="train", weighted_sampler = False):
        """
        Get the data loader

        Returns:
        - batch_x: list of tensors, each tensor is a batch of data for a modality
        - batch_y: tensor, batch of labels
        - missing_indices: list of indices of omitted modalities
        """
        if flag == 'train':
            shuffle_flag = True 
        else:
            shuffle_flag = False

        data  = data_set(self.args,data, flag)

        # if flag == "train":
        def collate_fn(batch):
            """
            
            """
            
            num_modalities = self.args.num_modalities
            miss_rate = self.args.miss_rate

            # Calculate the number of modalities to omit
            modalities_to_omit = ceil(miss_rate * num_modalities)
            batch_x = []
            batch_y  = []
            missing_indices = np.random.choice(num_modalities, modalities_to_omit, replace=False)

            # Collect batch
            for x, _, z in batch:
                
                batch_x.append(x)
                batch_y.append(z)

        
            batch_x = torch.tensor(np.concatenate(batch_x, axis=0))
            batch_y = torch.tensor(batch_y)
            
            # if np.random.uniform(0,1,1)[0] >= self.args.mixup_probability:

            #     batch_x , batch_y = mixup_data(batch_x , batch_y,   self.args.mixup_alpha,  argmax = self.args.mixup_argmax)
                #print("Mixup",batch_x1.shape,batch_y.shape)
            #else:
                #print(batch_x1.shape,batch_y.shape)
            batch_x = torch.unsqueeze(batch_x,1)

            if self.args.model_type == "shaspec":
                # Split the tensor into the number of modalities and convert to list
                batch_x = list(torch.tensor_split(batch_x, self.args.num_modalities, dim=3))

                # Setting the omitted modalities to NaN (or zero)
                for index in missing_indices:
                    # Ensure the tensor at the omitted modality index is filled with NaN
                    batch_x[index] = torch.full_like(batch_x[index], float('nan'))
            else:
                # Setting the omitted modalities to NaN (or zero)
                for index in missing_indices:
                    from_index = index * self.args.c_in
                    to_index = (index + 1) * self.args.c_in
                    # Ensure the tensor at the omitted modality index is filled with NaN
                    batch_x[:, :, :, from_index:to_index] = torch.full_like(batch_x[:, :, :, from_index:to_index], float('nan'))
                pass
            
            return batch_x, batch_y, missing_indices
        # else:
        #     collate_fn = None

        if weighted_sampler and flag == 'train':

            sampler = WeightedRandomSampler(
                data.act_weights, len(data.act_weights)
            )

            data_loader = DataLoader(data, 
                                     batch_size   =  self.args.batch_size,
                                     #shuffle      =  shuffle_flag,
                                     num_workers  =  0,
                                     sampler=sampler,
                                     drop_last    =  False,
                                     collate_fn = collate_fn)
        else:
            data_loader = DataLoader(data, 
                                     batch_size   =  self.args.batch_size,
                                     shuffle      =  shuffle_flag,
                                     num_workers  =  0,
                                     drop_last    =  False,
                                     collate_fn   = collate_fn)

        return data_loader


    def get_setting_name(self):
        if self.args.model_type == "shaspec":
            setting = "model_{}_data_{}_seed_{}_miss_rate_{}_use_shared_encoder_{}_use_missing_modality_features_{}".format(
                self.args.model_type,
                self.args.data_name,
                self.args.seed, 
                self.args.miss_rate, 
                self.args.use_shared_encoder,
                self.args.use_missing_modality_features 
                )
            return setting
        else:
            raise NotImplementedError
        # if self.args.model_type == "deepconvlstm":
        #     config_file = open('../../configs/model.yaml', mode='r')
        #     config = yaml.load(config_file, Loader=yaml.FullLoader)["deepconvlstm"]
        #     setting = "deepconvlstm_data_{}_seed_{}_windowsize_{}_waveFilter_{}_Fscaling_{}_cvfilter_{}_lstmfilter_{}_Regu_{}_wavelearnble_{}".format(self.args.data_name,
        #                                                                                                                                                 self.args.seed,
        #                                                                                                                                                 self.args.windowsize,
        #                                                                                                                                                 self.args.wavelet_filtering,
        #                                                                                                                                                 self.args.filter_scaling_factor,
        #                                                                                                                                                 config["nb_filters"],
        #                                                                                                                                                 config["nb_units_lstm"],
        #                                                                                                                                                 self.args.wavelet_filtering_regularization,
        #                                                                                                                                                 self.args.wavelet_filtering_learnable )
        #     return setting

        # if self.args.model_type == "deepconvlstm_attn":
        #     config_file = open('../../configs/model.yaml', mode='r')
        #     config = yaml.load(config_file, Loader=yaml.FullLoader)["deepconvlstm_attn"]
        #     setting = "deepconvlstm_attn_data_{}_seed_{}_windowsize_{}_waveFilter_{}_Fscaling_{}_cvfilter_{}_lstmfilter_{}_Regu_{}_wavelearnble_{}".format(self.args.data_name,
        #                                                                                                                                                   self.args.seed,
        #                                                                                                                                                   self.args.windowsize,
        #                                                                                                                                                   self.args.wavelet_filtering,
        #                                                                                                                                                   self.args.filter_scaling_factor,
        #                                                                                                                                                   config["nb_filters"],
        #                                                                                                                                                   config["nb_units_lstm"],
        #                                                                                                                                                   self.args.wavelet_filtering_regularization,
        #                                                                                                                                                   self.args.wavelet_filtering_learnable )
        #     return setting


        # if self.args.model_type == "mcnn":
        #     config_file = open('../../configs/model.yaml', mode='r')
        #     config = yaml.load(config_file, Loader=yaml.FullLoader)["mcnn"]
        #     setting = "mcnn_data_{}_seed_{}_windowsize_{}_waveFilter_{}_Fscaling_{}_cvfilter_{}_Regu_{}_wavelearnble_{}".format(self.args.data_name,
        #                                                                                                                                       self.args.seed,
        #                                                                                                                                       self.args.windowsize,
        #                                                                                                                                       self.args.wavelet_filtering,
        #                                                                                                                                       self.args.filter_scaling_factor,
        #                                                                                                                                       config["nb_filters"],
        #                                                                                                                                       self.args.wavelet_filtering_regularization,
        #                                                                                                                                       self.args.wavelet_filtering_learnable )
        #     return setting

        # elif self.args.model_type == "attend":
        #     config_file = open('../../configs/model.yaml', mode='r')
        #     config = yaml.load(config_file, Loader=yaml.FullLoader)["attend"]
        #     setting = "attend_data_{}_seed_{}_windowsize_{}_waveFilter_{}_Fscaling_{}_cvfilter_{}_grufilter_{}_Regu_{}_wavelearnble_{}".format(self.args.data_name,
        #                                                                                                                                        self.args.seed,
        #                                                                                                                                        self.args.windowsize,
        #                                                                                                                                        self.args.wavelet_filtering,
        #                                                                                                                                        self.args.filter_scaling_factor,
        #                                                                                                                                        config["filter_num"],
        #                                                                                                                                        config["hidden_dim"],
        #                                                                                                                                        self.args.wavelet_filtering_regularization,
        #                                                                                                                                        self.args.wavelet_filtering_learnable)
        #     return setting
        # elif self.args.model_type == "sahar":
        #     config_file = open('../../configs/model.yaml', mode='r')
        #     config = yaml.load(config_file, Loader=yaml.FullLoader)["sahar"]
        #     setting = "sahar_data_{}_seed_{}_windowsize_{}_waveFilter_{}_Fscaling_{}_cvfilter_{}_grufilter_{}_Regu_{}_wavelearnble_{}".format(self.args.data_name,
        #                                                                                                                                       self.args.seed,
        #                                                                                                                                       self.args.windowsize,
        #                                                                                                                                       self.args.wavelet_filtering,
        #                                                                                                                                       self.args.filter_scaling_factor,
        #                                                                                                                                       config["nb_filters"],
        #                                                                                                                                       None,
        #                                                                                                                                       self.args.wavelet_filtering_regularization,
        #                                                                                                                                       self.args.wavelet_filtering_learnable)
        #     return setting
        # elif self.args.model_type == "tinyhar":
        #     config_file = open('../../configs/model.yaml', mode='r')
        #     config = yaml.load(config_file, Loader=yaml.FullLoader)["tinyhar"]
        #     setting = "tinyhar_data_{}_seed_{}_windowsize_{}_cvfilter_{}_CI_{}_CA_{}_TI_{}_TA_{}".format(self.args.data_name,
        #                                                                                                 self.args.seed,
        #                                                                                                 self.args.windowsize,
        #                                                                                                 config["filter_num"],
        #                                                                                                 self.args.cross_channel_interaction_type,
        #                                                                                                 self.args.cross_channel_aggregation_type,
        #                                                                                                 self.args.temporal_info_interaction_type,
        #                                                                                                 self.args.temporal_info_aggregation_type )
        #     return setting
        # else:
        #     raise NotImplementedError



    def update_gamma(self ):
        for n, parameter in self.model.named_parameters():
            if "gamma" in n:
                parameter.grad.data.add_(self.args.regulatization_tradeoff*torch.sign(parameter.data))  # L1


    def train(self):
        start_time = time.time()

        setting = self.get_setting_name()

        path = os.path.join(self.args.to_save_path,'logs/'+setting)
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

        score_log_file_name = os.path.join(self.path, "score.txt")
        config_file_name = os.path.join(self.path, "config.json")

        config_dict = {
            'model_type': self.args.model_type,
            'model_size': int(self.args.model_size),
            'data_name': self.args.data_name,
            'seed': self.args.seed,
            'batch_size': self.args.batch_size,
            'shuffle': self.args.shuffle,
            'drop_last': self.args.drop_last,
            'train_vali_quote': self.args.train_vali_quote,
            'train_epochs': self.args.train_epochs,
            'learning_rate': self.args.learning_rate,
            'learning_rate_patience': self.args.learning_rate_patience,
            'learning_rate_factor': self.args.learning_rate_factor,
            'early_stop_patience': self.args.early_stop_patience,
            'weight_decay': self.args.weight_decay,
            'use_gpu': self.args.use_gpu,
            'gpu': self.args.gpu,
            'use_multi_gpu': self.args.use_multi_gpu,
            'optimizer': self.args.optimizer,
            'criterion': self.args.criterion,
            'activation': self.args.activation,
            'shared_encoder_type': self.args.shared_encoder_type,
            'use_shared_encoder': self.args.use_shared_encoder,
            'use_missing_modality_features': self.args.use_missing_modality_features,
            'sampling_freq': self.args.sampling_freq,
            'num_classes': self.args.num_classes,
            'num_modalities': self.args.num_modalities,
            'miss_rates': self.args.miss_rates,
            'selected miss_rate': self.args.miss_rate,
            'windowsize': self.args.windowsize,
            'input_length': self.args.input_length,
            'c_in_per_mod': self.args.c_in_per_mod,
            'f_in': self.args.f_in,
        }

        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True 
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        # Load the data
        dataset = data_dict[self.args.data_name](self.args)

        print("=" * 16, f" {dataset.exp_mode} Mode ", "=" * 16)
        print("=" * 16, f" {dataset.num_of_cv} CV folds in total ", "=" * 16)

        num_of_cv = dataset.num_of_cv

        # Write the configuration settings to a file
        with open(config_file_name, 'w') as config_file:
            json.dump(config_dict, config_file, indent=4)

        # Initialize a list to hold scores for each fold
        cv_scores = []

        for iter in range(num_of_cv):
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            torch.backends.cudnn.deterministic = True 
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            g = torch.Generator()
            g.manual_seed(self.args.seed)                  
            torch.backends.cudnn.benchmark = False
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

            print("=" * 16, f" CV {iter} ", "=" * 16)
    
            dataset.update_train_val_test_keys()

            cv_path = os.path.join(self.path, f"cv_{iter}")

            # Get the loader of train val test
            train_loader = self._get_data(dataset, flag = 'train', weighted_sampler = self.args.weighted_sampler)
            val_loader = self._get_data(dataset, flag = 'vali', weighted_sampler = self.args.weighted_sampler)
            test_loader   = self._get_data(dataset, flag = 'test', weighted_sampler = self.args.weighted_sampler)
            #class_weights=torch.tensor(dataset.act_weights,dtype=torch.double).to(self.device)
            train_steps = len(train_loader)

            if not os.path.exists(cv_path):
                os.makedirs(cv_path)
                skip_train = False
                skip_finetuning = False
            else:
                file_in_folder = os.listdir(cv_path)
                if 'final_best_vali.pth' in file_in_folder:
                    skip_train = True
                else:
                    skip_train = False

                if 'final_finetuned_best_vali.pth' in file_in_folder:
                    skip_finetuning = True
                else:
                    skip_finetuning = False

            epoch_log_file_name = os.path.join(cv_path, "epoch_log.txt")

            if skip_train:
                print("=" * 16, f" Skip the {iter} CV experiment ", "=" * 16)
            else:

                if os.path.exists(epoch_log_file_name):
                    os.remove(epoch_log_file_name)

                epoch_log = open(epoch_log_file_name, "a")
                score_log = open(score_log_file_name, "a")

                print("=" * 16, f" Build the {self.args.model_type} model ", "=" * 16)	
        
                self.model  = self.build_model().to(self.device)

                early_stopping        = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)
                learning_rate_adapter = adjust_learning_rate_class(self.args,True)
                model_optim = self._select_optimizer()

                #if self.args.weighted == True:
                #    criterion =  nn.CrossEntropyLoss(reduction="mean",weight=class_weights).to(self.device)#self._select_criterion()
                #else:
                #    criterion =  nn.CrossEntropyLoss(reduction="mean").to(self.device)#self._select_criterion()
                criterion =  nn.CrossEntropyLoss(reduction="mean").to(self.device)
                criterion = MixUpLoss(criterion)

                for epoch in tqdm(range(self.args.train_epochs), desc='Training Process [epochs]'):
                    train_loss = []
                    self.model.train()
                    epoch_time = time.time()
                
                    for (batch_x,batch_y,missing_indices) in train_loader:                        
                        # Ensure models and variables are on the same device
                        if self.args.model_type == "shaspec":
                            batch_x = [x.double().to(self.device) for x in batch_x]
                        else:
                            batch_x = batch_x.double().to(self.device)

                        batch_y = batch_y.long().to(self.device)
                        
                        if self.args.model_type == "shaspec":
                            outputs = self.model(batch_x, missing_indices)
                        else:
                            outputs = self.model(batch_x)
                        
                        loss = criterion(outputs, batch_y)
    
                        if self.args.wavelet_filtering and self.args.wavelet_filtering_regularization:
                            reg_loss = 0
                            for name,parameter in self.model.named_parameters():
                                if "gamma" in name:
                                    reg_loss += torch.sum(torch.abs(parameter))

                            loss = loss + self.args.regulatization_tradeoff*reg_loss

                        train_loss.append(loss.item())

                        model_optim.zero_grad()
                        loss.backward()
                        model_optim.step()


                    print(f"Epoch: {epoch + 1} cost time: {time.time()-epoch_time}")
                    epoch_log.write("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
                    epoch_log.write("\n")

                    train_loss = np.average(train_loss)
                    vali_loss , vali_acc, vali_f_w,  vali_f_macro,  vali_f_micro = self.validation(self.model, val_loader, criterion)

                    print("VALI: Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Vali Accuracy: {4:.7f}  Vali weighted F1: {5:.7f}  Vali macro F1 {6:.7f} ".format(
                        epoch + 1, train_steps, train_loss, vali_loss, vali_acc, vali_f_w, vali_f_macro))

                    epoch_log.write("VALI: Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Vali Accuracy: {4:.7f}  Vali weighted F1: {5:.7f}  Vali macro F1 {6:.7f} \n".format(
                        epoch + 1, train_steps, train_loss, vali_loss, vali_acc, vali_f_w, vali_f_macro))

                    early_stopping(vali_loss, self.model, cv_path, vali_f_macro, vali_f_w, epoch_log)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                    epoch_log.write("----------------------------------------------------------------------------------------\n")
                    epoch_log.flush()
                    learning_rate_adapter(model_optim,vali_loss)
            
                # Rename the best_vali to final_best_vali
                os.rename(cv_path+'/'+'best_vali.pth', cv_path+'/'+'final_best_vali.pth')

                print("Loading the best validation model!")
                self.model.load_state_dict(torch.load(cv_path+'/'+'final_best_vali.pth'))
                #model.eval()
                test_loss , test_acc, test_f_w,  test_f_macro,  test_f_micro = self.validation(self.model, test_loader, criterion, iter+1)

                cv_scores.append((test_loss, test_acc, test_f_w, test_f_macro, test_f_micro))

                print(f"Final Test Performance : Test Accuracy: {test_acc:.7f}  Test weighted F1: {test_f_w:.7f}  Test macro F1 {test_f_macro:.7f}")
                epoch_log.write(f"Final Test Performance : Test Loss: {test_loss:.7f}  Test Accuracy: {test_acc:.7f}  Test weighted F1: {test_f_w:.7f}  Test macro F1 {test_f_macro:.7f}  Test micro F1: {test_f_micro:.7f}\n")
                epoch_log.flush()

                score_log.write(f"Test Loss: {test_loss:.7f}  Test Accuracy: {test_acc:.7f}  Test weighted F1: {test_f_w:.7f}  Test macro F1 {test_f_macro:.7f}  Test micro F1: {test_f_micro:.7f}\n")
                score_log.flush()

                epoch_log.close()
                score_log.close()

            # ------------------------------ code for  regularization and fine tuning -----------------------------------------------------------------

            if self.args.wavelet_filtering_finetuning:
                finetuned_score_log_file_name = os.path.join(self.path, "finetuned_score.txt")
                if skip_finetuning:
                    print("=" * 16, " Skip the {iter} CV Experiment Fine Tuning ", "=" * 16)
                else:
                    # thre_index : selected number
                    epoch_log = open(epoch_log_file_name, "a")
                    epoch_log.write("----------------------------------------------------------------------------------------\n")
                    epoch_log.write("--------------------------------------Fine Tuning-----------------------------------------\n")
                    epoch_log.write("----------------------------------------------------------------------------------------\n")

                    self.model  = self.build_model().to(self.device)
                    self.model.load_state_dict(torch.load(cv_path+'/'+'final_best_vali.pth'))

                    finetuned_score_log = open(finetuned_score_log_file_name, "a")

                    thre_index             = int(self.args.f_in * self.args.wavelet_filtering_finetuning_percent)-1
                    gamma_weight           = self.model.gamma.squeeze().abs().clone()
                    sorted_gamma_weight, i = torch.sort(gamma_weight,descending=True)
                    threshold              = sorted_gamma_weight[thre_index]
                    mask                   = gamma_weight.data.gt(threshold).float().to(self.device)
                    idx0                   = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    # build the new model
                    new_model              = model_builder(self.args, input_f_channel = thre_index).to(self.device)

                    print("=" * 16, " Fine Tuning  : ", self.args.f_in-thre_index,"  will be pruned ", "=" * 16)
                    print("Old model parameter :", self.model_size)
                    print("Pruned model parameter :", np.sum([para.numel() for para in new_model.parameters()]))
                    print("=" * 64)

                    # Copy the weights
                    flag_channel_selection = False
                    for n,p in new_model.named_parameters():
                        if "wavelet_conv" in n:
                            p.data = self.model.state_dict()[n].data[idx0.tolist(), :,:,:].clone()
                        elif n == "gamma":
                            flag_channel_selection = True
                            p.data = self.model.state_dict()[n].data[:, idx0.tolist(),:,:].clone()
                        elif flag_channel_selection and "conv" in n:
                            p.data = self.model.state_dict()[n].data[:, idx0.tolist(),:,:].clone()
                            flag_channel_selection = False
                        else:
                            p.data = self.model.state_dict()[n].data.clone()

                    early_stopping        = EarlyStopping(patience=15, verbose=True)
                    learning_rate_adapter = adjust_learning_rate_class(self.args,True)
                    model_optim           = optim.Adam(new_model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
                    criterion             = nn.CrossEntropyLoss(reduction="mean").to(self.device)
                    for epoch in range(self.args.train_epochs):
                        train_loss = []
                        new_model.train()
                        epoch_time = time.time()

                        for (batch_x,batch_y,missing_indices) in train_loader:
                            # batch_x = batch_x.double().to(self.device)
                            batch_x = [x.double().to(self.device) for x in batch_x]
                            batch_y = batch_y.long().to(self.device)

                            
                            outputs = new_model(batch_x, missing_indices)
                            

                            loss = criterion(outputs, batch_y)

                            train_loss.append(loss.item())

                            model_optim.zero_grad()
                            loss.backward()
                            model_optim.step()

                        print(f"Fine Tuning Epoch: {epoch+1} cost time: {time.time()-epoch_time}")
                        epoch_log.write("Fine Tuning Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
                        epoch_log.write("\n")

                        train_loss = np.average(train_loss)
                        vali_loss , vali_acc, vali_f_w,  vali_f_macro, vali_f_micro = self.validation(new_model, val_loader, criterion)

                        print(f"Fine Tuning VALI: Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Vali Accuracy: {vali_acc:.7f} Vali weighted F1: {vali_f_w:.7f} Vali macro F1 {vali_f_macro:.7f}")

                        epoch_log.write("Fine Tuning VALI: Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Vali Accuracy: {4:.7f}  Vali weighted F1: {5:.7f}  Vali macro F1 {6:.7f} \n".format(
                            epoch + 1, train_steps, train_loss, vali_loss, vali_acc, vali_f_w, vali_f_macro))

                        early_stopping(vali_loss, new_model, cv_path, vali_f_macro, vali_f_w, epoch_log)
                        if early_stopping.early_stop:
                            print("Early stopping")
                            break
                        epoch_log.write("----------------------------------------------------------------------------------------\n")
                        epoch_log.flush()
                        learning_rate_adapter(model_optim,vali_loss)
                    # rename the best_vali to final_best_vali
                    os.rename(cv_path+'/'+'best_vali.pth', cv_path+'/'+'final_finetuned_best_vali.pth')

                    print("Loading the best finetuned validation model!")
                    new_model.load_state_dict(torch.load(cv_path+'/'+'final_finetuned_best_vali.pth'))

                    test_loss , test_acc, test_f_w,  test_f_macro,  test_f_micro = self.validation(new_model, test_loader, criterion)
                    print(f"Fine Tuning Final Test Performance : Test Accuracy: {test_acc:.7f}  Test weighted F1: {test_f_w:.7f}  Test macro F1 {test_f_macro:.7f}")
                    epoch_log.write("Final Test Performance : Test weighted F1: {0:.7f}  Test macro F1 {1:.7f}\n\n\n\n\n\n\n\n".format(test_f_w, test_f_macro))
                    epoch_log.flush()

                    finetuned_score_log.write("Test weighted F1: {0:.7f}  Test macro F1 {1:.7f}\n".format(test_f_w, test_f_macro))
                    finetuned_score_log.flush()

                    epoch_log.close()
                    finetuned_score_log.close()
        end_time = time.time()
        training_duration_hours = (end_time - start_time) / 3600

        # Load the existing config
        with open(config_file_name, 'r') as config_file:
            config_dict = json.load(config_file)

        # Update the config with the training duration
        config_dict['training_duration_hours'] = training_duration_hours

        # Save the updated config
        with open(config_file_name, 'w') as config_file:
            json.dump(config_dict, config_file, indent=4)

        cv_scores = np.array(cv_scores)

        if cv_scores.size > 0:
            # Calculate mean and standard deviation for each metric, including test_loss
            mean_scores = np.mean(cv_scores, axis=0)
            std_scores = np.std(cv_scores, axis=0)

            with open(score_log_file_name, "a") as score_log:
                score_log.write("\n")
                score_log.write(f"MEAN Test Loss: {mean_scores[0]:.7f}, Test Accuracy: {mean_scores[1]:.7f}, Test weighted F1: {mean_scores[2]:.7f}, Test macro F1: {mean_scores[3]:.7f}, Test micro F1: {mean_scores[4]:.7f}\n")
                score_log.write(f"STD Test Loss: {std_scores[0]:.7f}, Test Accuracy: {std_scores[1]:.7f}, Test weighted F1: {std_scores[2]:.7f}, Test macro F1: {std_scores[3]:.7f}, Test micro F1: {std_scores[4]:.7f}\n")
        

    def prediction_test(self):
        print("PREDICTION TEST")
        assert self.args.exp_mode == "Given"
        model = self.build_model().to(self.device)
        model.load_state_dict(torch.load(os.path.join(self.path,'cv_0/best_vali.pth')))
        model.eval()
        dataset = data_dict[self.args.data_name](self.args)
        dataset.update_train_val_test_keys()
        test_loader   = self._get_data(dataset, flag = 'test')
        preds = []
        trues = []
        for i, (batch_x1,batch_x2,batch_y) in enumerate(test_loader):
            if "cross" in self.args.model_type:
                batch_x1 = batch_x1.double().to(self.device)
                batch_x2 = batch_x2.double().to(self.device)
                batch_y = batch_y.long().to(self.device)
                # model prediction
                if self.args.output_attention:
                    outputs = self.model(batch_x1,batch_x2)[0]
                else:
                    outputs = self.model(batch_x1,batch_x2)
            else:
                batch_x1 = batch_x1.double().to(self.device)
                batch_y = batch_y.long().to(self.device)

                # model prediction
                if self.args.output_attention:
                    outputs = self.model(batch_x1)[0]
                else:
                    outputs = self.model(batch_x1)

            preds.extend(list(np.argmax(outputs.detach().cpu().numpy(),axis=1)))
            trues.extend(list(batch_y.detach().cpu().numpy())) 
		
        acc = accuracy_score(preds,trues)
        f_w = f1_score(trues, preds, average='weighted')
        f_macro = f1_score(trues, preds, average='macro')
        f_micro = f1_score(trues, preds, average='micro')

        return preds,trues

    def validation(self, model, data_loader, criterion, index_of_cv=None, selected_index = None):
        model.eval()
        total_loss = []
        preds = []
        trues = []
        with torch.no_grad():
            for (batch_x,batch_y,missing_indices) in data_loader:
                if selected_index is None:
                    batch_x = [x.double().to(self.device) for x in batch_x]
                else:
                    batch_x = batch_x[:, selected_index.tolist(),:,:].double().to(self.device)
                batch_y = batch_y.long().to(self.device)

                # model prediction
                
                outputs = model(batch_x, missing_indices)
                

                pred = outputs.detach()#.cpu()
                true = batch_y.detach()#.cpu()

                loss = criterion(pred, true) 
                total_loss.append(loss.cpu())
				
                preds.extend(list(np.argmax(outputs.detach().cpu().numpy(),axis=1)))
                trues.extend(list(batch_y.detach().cpu().numpy()))   
				
        total_loss = np.average(total_loss)
        acc = accuracy_score(preds,trues)
        #f_1 = f1_score(trues, preds)
#         print("------------------11111111111111---------------")
#         import pickle
#         with open('preds.pickle', 'wb') as handle:
#             pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
#         with open('trues.pickle', 'wb') as handle:
#             pickle.dump(trues, handle, protocol=pickle.HIGHEST_PROTOCOL)
          
        f_w = f1_score(trues, preds, average='weighted')
        f_macro = f1_score(trues, preds, average='macro')
        f_micro = f1_score(trues, preds, average='micro')
        if index_of_cv:
            cf_matrix = confusion_matrix(trues, preds)
            #with open("{}.npy".format(index_of_cv), 'wb') as f:
            #    np.save(f, cf_matrix)
            plt.figure()
            sns.heatmap(cf_matrix, annot=True)
            #plt.savefig("{}.png".format(index_of_cv))
        model.train()

        return total_loss,  acc, f_w,  f_macro, f_micro#, f_1
