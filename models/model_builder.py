# ---- Import models ---------------
from models.ShaSpec import ShaSpec
from models.Attend import AttendDiscriminate
from dataloaders.utils import PrepareWavelets,FiltersExtention
# ------- Import other packages ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class model_builder(nn.Module):
    """
    Builds the model based on the arguments
    """
    def __init__(self, args, input_f_channel = None):
        super(model_builder, self).__init__()

        self.args = args
        if input_f_channel is None:
            f_in  = self.args.f_in
        else:
            f_in  = input_f_channel
        
        if self.args.model_type == "shaspec":
            config_file = open('configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["shaspec"]
            self.model  = ShaSpec((1, f_in, self.args.input_length, self.args.c_in_per_mod),
                                self.args.num_modalities,
                                self.args.miss_rate,
                                self.args.num_classes,
                                self.args.activation,
                                self.args.shared_encoder_type, # concatenated weighted
                                self.args.use_shared_encoder, # Ablation study 
                                self.args.use_missing_modality_features,  # Ablation study
                                config)
            
            print("Build the ShaSpec model!")
        else:
            self.model = Identity()
            print("Build the None model!")


    def forward(self, x, missing_indices=None):        
        y = self.model(x, missing_indices)
        
        return y
    