#!/usr/bin/env python

#SBATCH --job-name=shaspec

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
# SDIL path
sys.path.append('/pfs/data5/home/kit/tm/px6680/lukasprobst/shaspec-har/')
import argparse
from experiment import Exp

import torch
import yaml

parser = argparse.ArgumentParser()

# Save info config
parser.add_argument('--to-save-path', dest='to_save_path', default= "../Run_logs", type=str, help='Set the path to save logs')
parser.add_argument('--freq-save-path', dest='freq_save_path', default= "../Freq_data", type=str, help='Set the path to save freq images transformation')
parser.add_argument('--window-save-path', dest='window_save_path', default= "../Sliding_window", type=str, help='Set the path to save slided window file')
parser.add_argument('--root-path', dest='root_path', default= "../datasets", type=str, help='Set the path to data dir')

# Data normalization
parser.add_argument('--datanorm-type', dest='datanorm_type', default= "standardization", choices=[None, "minmax", "standardization"],
                    type=str, help='Set the method to standize the data')
parser.add_argument('--sample-wise', dest='sample_wise', action='store_true', help='Whether to perform sample_wise normailization')

parser.add_argument('--drop-transition', dest='drop_transition', action='store_true', help='Whether to drop the transition part')

# Training config
parser.add_argument('--batch-size', dest='batch_size', default=128, type=int,  help='Set the batch size')
parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='Whether to shuffle the data')
parser.add_argument('--drop-last', dest='drop_last', action='store_true', help='Whether to drop the last mini batch')
parser.add_argument('--train-vali-quote', dest='train_vali_quote', type=float, default=0.9, help='Portion of training dataset')
parser.add_argument('--train-epochs', dest='train_epochs', default=60, type=int,  help='Total training Epochs')
parser.add_argument('--learning-rate', dest='learning_rate', default=0.001, type=float,  help='Set the initial learning rate')
parser.add_argument('--learning-rate-patience', dest='learning_rate_patience', default=20, type=int,  help='Patience for adjust the learning rate')
parser.add_argument('--early-stop-patience', dest='early_stop_patience', default=15, type=int,  help='Patience for stop the training')
parser.add_argument('--learning-rate-factor', dest='learning_rate_factor', default=0.1, type=float,  help='Set the rate of adjusting learning rate')
parser.add_argument('--weight-decay', dest='weight_decay', default=0, type=float,  help='Set the weight decay')
parser.add_argument('--use-multi-gpu', dest='use_multi_gpu', action='store_true', help='Whether to use multi gpu')
parser.add_argument('--gpu', dest='gpu', default=0, type=int,  help='Set the gpu id')
parser.add_argument('--optimizer', dest='optimizer', default= "Adam", type=str, help='Set the optimized type')
parser.add_argument('--criterion', dest='criterion', default= "CrossEntropy", type=str, help='Set the loss type')
parser.add_argument('--seed', dest='seed', default=1, type=int,  help='Set the training seed')

parser.add_argument('--data-name', dest='data_name', default= None, type=str, help='Set the dataset name')

parser.add_argument('--difference', dest='difference', action='store_true', help='Whether to use difference as input')
parser.add_argument('--filtering', dest='filtering', action='store_true', help='Whether to use filtering as input')
parser.add_argument('--load-all', dest='load_all', action='store_true', help='Whether to load all freq data')

# Experiment mode
parser.add_argument('--exp-mode', dest='exp_mode', default= "LOCV", type=str, help='Set the experiment type')

parser.add_argument('--model-type', dest='model_type', default= None, type=str, help='Set the model type')
parser.add_argument('--filter-scaling-factor', dest='filter_scaling_factor', default=1.0, type=float,  help='Set the scaling factor for filter dimension')

# ShaSpec-specific
parser.add_argument('--activation', dest='activation', default= "ReLU", type=str, help='Set the activation function')
parser.add_argument('--shared-encoder-type', dest='shared_encoder_type', default= "concatenated", type=str, help='Set the shared_encoder_type type for the ShaSpec model')
parser.add_argument('--miss-rate', dest='miss_rate', default=0.0, type=float, help='Set the miss rate for modalities')
parser.add_argument('--ablate-shared-encoder', dest='ablate_shared_encoder', action='store_true', help='Whether the shared encoder should be ablated or not')
parser.add_argument('--ablate-missing-modality-features', dest='ablate_missing_modality_features', action='store_true', help='Whether the missing modality feature generation should be ablated or not')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() else False

args.pos_select       = None
args.sensor_select    = None

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
# Input information
# args.c_in            =  config["num_channels"]
# For ShaSpec we want the number of channels per modality
args.c_in    =  config["num_channels"] // args.num_modalities

if args.difference:
    args.c_in  = args.c_in * 2
if  args.filtering :
    for col in config["sensors"]:
        if "acc" in col:
            args.c_in = args.c_in + 1

args.f_in            =  1

print("Configuration done.")

exp = Exp(args)
exp.train()

print("Training done.")
