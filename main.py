#!/usr/bin/env python

#SBATCH --job-name=shaspec24

#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

#SBATCH --mail-user=probst@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil
#SBATCH --gres=gpu:1

import os
import sys

sys.path.append(os.getcwd())
sys.path.append('/pfs/data5/home/kit/tm/px6680/lukasprobst/I2S0W2C2_CFC/')
import argparse
from experiment import Exp

from dataloaders import data_set, data_dict
import torch
import yaml

parser = argparse.ArgumentParser()

# save info config
parser.add_argument('--to-save-path', dest='to_save_path', default= "../Run_logs", type=str, help='Set the path to save logs')
parser.add_argument('--freq-save-path', dest='freq_save_path', default= "../Freq_data", type=str, help='Set the path to save freq images transformation')
parser.add_argument('--window-save-path', dest='window_save_path', default= "../Sliding_window", type=str, help='Set the path to save slided window file')
parser.add_argument('--root-path', dest='root_path', default= "../datasets", type=str, help='Set the path to data dir')

# data normalization
parser.add_argument('--datanorm-type', dest='datanorm_type', default= "standardization", choices=[None, "minmax", "standardization"],
                    type=str, help='Set the mathod for standize the data')
parser.add_argument('--sample-wise', dest='sample_wise', action='store_true', help='Whether to perform sample_wise normailization')


parser.add_argument('--drop-transition', dest='drop_transition', action='store_true', help='Wether to drop the transition part')

# training config
parser.add_argument('--batch-size', dest='batch_size', default=256, type=int,  help='Set the batch size')
parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='Whether to shuffle the data')
parser.add_argument('--drop-last', dest='drop_last', action='store_true', help='Whether to drop the last mini batch')
parser.add_argument('--train-vali-quote', dest='train_vali_quote', type=float, default=0.9, help='Portion of training dataset')
parser.add_argument('--train-epochs', dest='train_epochs', default=60, type=int,  help='Total Training Epochs')
parser.add_argument('--learning-rate', dest='learning_rate', default=0.001, type=float,  help='set the initial learning rate')
parser.add_argument('--learning-rate-patience', dest='learning_rate_patience', default=20, type=int,  help='patience for adjust the learning rate')
parser.add_argument('--early-stop-patience', dest='early_stop_patience', default=15, type=int,  help='patience for stop the training')
parser.add_argument('--learning-rate-factor', dest='learning_rate_factor', default=0.1, type=float,  help='set the rate of adjusting learning rate')
parser.add_argument('--weight-decay', dest='weight_decay', default=0.01, type=float,  help='Set the weight decay')
parser.add_argument('--use-multi-gpu', dest='use_multi_gpu', action='store_true', help='Whether to use multi gpu ')
parser.add_argument('--gpu', dest='gpu', default=0, type=int,  help='gpu id')
parser.add_argument('--optimizer', dest='optimizer', default= "Adam", type=str, help='Set the Optimized Type')
parser.add_argument('--criterion', dest='criterion', default= "CrossEntropy", type=str, help='Set the Loss Type')
parser.add_argument('--seed', dest='seed', default=1, type=int,  help='Set the training seed')

parser.add_argument('--data-name', dest='data_name', default= None, type=str, help='Set the dataset name')

parser.add_argument('--weighted-sampler', dest='weighted_sampler', action='store_true', help='Whether to use weighted_sampler')
parser.add_argument('--mixup-alpha', dest='mixup_alpha', default=0.5, type=float,  help='Set the mixup distribution')

parser.add_argument('--representation-type', dest='representation_type', default= "time", type=str, help='Set the data type')
# Experiment mode
parser.add_argument('--exp-mode', dest='exp_mode', default= "LOCV", type=str, help='Set the exp type')

parser.add_argument('--model-type', dest='model_type', default= None, type=str, help='Set the model type')
parser.add_argument('--output-attention', dest='output_attention', action='store_true', help='Whether to print out the attention map if cross attention model is used')
parser.add_argument('--filter-scaling-factor', dest='filter_scaling_factor', default=1.0, type=float,  help='Set the scaling factor for filter dimension')

# ShaSpec-specific
parser.add_argument('--activation', dest='activation', default= "ReLU", type=str, help='Set the activation function')
parser.add_argument('--shared-encoder-type', dest='shared_encoder_type', default= "concatenated", type=str, help='Set the shared_encoder_type type for the ShaSpec model')
parser.add_argument('--miss-rate', dest='miss_rate', default=0.0, type=float, help='Set the miss rate for modalities')
parser.add_argument('--use-shared-encoder', dest='use_shared_encoder', default=True, type=bool, help='Whether the shared encoder should be used or not')
parser.add_argument('--use-missing-modality-features', dest='use_missing_modality_features', default=True, type=bool, help='Whether the missing modality feature generation should be used or not')

args = parser.parse_args()

args.use_gpu                 = True if torch.cuda.is_available() else False

args.random_augmentation_config = {"jitter":True,
                                   "moving_average":True,
                                   "magnitude_scaling":True,
                                   "magnitude_warp":True,
                                   "magnitude_shift":True,
                                   "time_warp":True,
                                   "window_warp":True,
                                   "window_slice":True,
                                   "random_sampling":True,
                                   "slope_adding":True
                                   }
random_augmentation_nr = 0
for key in args.random_augmentation_config.keys():
    if args.random_augmentation_config[key]:
        random_augmentation_nr = random_augmentation_nr+1
args.random_augmentation_nr = random_augmentation_nr


args.pos_select       = None
args.sensor_select    = None

if args.data_name ==  "skodar":
    args.exp_mode = "SOCV"

config_file = open('configs/data.yaml', mode='r')
data_config = yaml.load(config_file, Loader=yaml.FullLoader)
config = data_config[args.data_name]

args.root_path       =  os.path.join(args.root_path,config["filename"])
args.sampling_freq   =  config["sampling_freq"]
args.num_classes     =  config["num_classes"]
args.num_modalities  =  config["num_modalities"]
args.miss_rates      =  config["miss_rates"]
window_seconds       =  config["window_seconds"]
args.windowsize      =  int(window_seconds * args.sampling_freq) 
args.input_length    =  args.windowsize
# input information
args.c_in            =  config["num_channels"]
# For ShaSpec we want the number of channels per modality
args.c_in_per_mod    =  config["num_channels"] // args.num_modalities

args.f_in            =  1

print("Configuration done.")

exp = Exp(args)
exp.train()
print("Training done.")
