import torch # For all things PyTorch
import torch.nn as nn # For torch.nn.Module, the parent object for PyTorch models
from models.Attend import SelfAttention # For the self-attention mechanism
from math import ceil # For rounding up the number of available modalities


class BaseEncoder(nn.Module):
    def __init__(self, input_shape, filter_num, filter_size, activation, sa_div):
        super(BaseEncoder, self).__init__()
        
        """
        PART 1: Channel-wise Feature Extraction (Convolutional layers)
        
        Input: batch_size, filter_num, temp_length, num_of_sensor_channels
        Output: batch_size, filter_num, downsampled_length, num_of_sensor_channels
        """
        # Halving the length with each convolutional layer in the sequential container
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[1], filter_num, (filter_size, 1), stride = (2, 1)),
            self.get_activation(activation),
            nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride = (2, 1)),
            self.get_activation(activation),
            nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride = (2, 1)),
            self.get_activation(activation),
            nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride = (2, 1)),
            self.get_activation(activation)
        )

        """
        PART 2: Cross-Channel Interaction (Self-Attention)

        Input: batch_size, filter_num, downsampled_length, num_of_sensor_channels
        Output: batch_size, filter_num, downsampled_length, num_of_sensor_channels
        """
        self.sa = SelfAttention(filter_num, sa_div)

        """
        PART 3: Cross-Channel Fusion (FC layer)

        Input: batch_size, num_of_sensor_channels, downsampled_length * filter_num
        Output: batch_size, num_of_sensor_channels, 2 * filter_num
        """
        downsampled_length = self.get_downsampled_length(input_shape)

        self.fc_fusion = nn.Linear(downsampled_length * filter_num, 2 * filter_num)

    def forward(self, x):
        """
        Perform a forward pass of the SpecificEncoder model.

        Args:
            x (Tensor): The input tensor of shape (B=batch_size, F=filter_num, T=temp_length, C=num_of_sensor_channels).
        
        Returns:
            Tensor: The output tensor after applying convolutions, self-attention, and a fully connected layer.
        
        The function applies a series of convolutional layers to extract features from each sensor channel, followed by a self-attention mechanism to refine these features. 
        Finally, it reshapes and passes the output through a fully connected layer for sensor channel fusion.
        """
        # B x F x T x C
        x = self.conv_layers(x)
        # -----> B x F' x T* x C

        """ ================ Cross-Channel Interaction ================"""
        # Apply self-attention to each time step in the sequence
        # For each time step 't' in the sequence:
        # 1. Extract the slice of data corresponding to the time step.
        # 2. Apply self-attention to this slice. The self-attention mechanism refines
        #    the features by considering the interaction of this time step with all other time steps in the sequence.
        # 3. Use torch.unsqueeze to ensure the slice has the correct dimensionality for self-attention.
        # Finally, concatenate the refined time steps along the last dimension to reconstruct the sequence with enhanced features.
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        # -----> B x F' x C x T*

        """ ================ Cross-Channel Fusion ================"""
        x = refined.permute(0, 2, 1, 3)
        # -----> B x C x F' x T*

        # Flatten the last two dimensions
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # -----> B x C x (T* F')

        # Pass through FC layer
        x = self.fc_fusion(x)
        # -----> B x C x 2F'
        return x

    @staticmethod
    def get_activation(activation_type):
        if activation_type == "ReLU":
            return nn.ReLU()
        else:
            return nn.Tanh()

    def get_downsampled_length(self, input_shape):
        """
        Compute the downsampled length after the convolutional layers.
        """
        # No need to track gradients here
        with torch.no_grad():  
            x = torch.randn(input_shape)
            # Pass through all conv layers at once
            x = self.conv_layers(x)  
            # Return the temporal length after downsampling
            return x.shape[2]  


class SpecificEncoder(nn.Module):
    """
    Specific Encoder to capture the unique characteristics of each modality.

    Input: modality
    Output: specific feature
    """
    def __init__(self, input_shape, filter_num, filter_size, activation, sa_div):
        super(SpecificEncoder, self).__init__()

        self.be = BaseEncoder(
            input_shape,
            filter_num,
            filter_size,
            activation,
            sa_div
        )

    def forward(self, x):
        x = self.be(x)

        return x


class SharedEncoder(nn.Module):
    """
    Shared Encoder to to extract common features that are informative accross all sensor channels.

    Input: N concatenated modalities
    Output: N shared features as output
    """
    def __init__(self, num_available_modalities, input_shape, filter_num, filter_size, activation, sa_div, shared_encoder_type):
        super(SharedEncoder, self).__init__()

        self.num_available_modalities = num_available_modalities

        # Concatenate modalities
        input_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] * num_available_modalities)

        self.be = BaseEncoder(
            input_shape,
            filter_num,
            filter_size, 
            activation, 
            sa_div
        )
        
    def forward(self, x):
        x = self.be(x)       
        shared_features = torch.tensor_split(x, self.num_available_modalities, dim=1)
        return shared_features


class ResidualBlock(nn.Module):
    """
    Residual Block to fuse specific and shared features.

    Input: specific feature, shared feature
    Output: modality embedding
    """
    def __init__(
        self,
        filter_num,
    ):
        super(ResidualBlock, self).__init__()

        # Linear projection
        self.fc_proj = nn.Linear(4 * filter_num, 2 * filter_num)

    def forward(self, specific_feature, shared_feature):
        # Concatenate
        concatenated = torch.cat((specific_feature, shared_feature), dim=2)

        # Linear projection for fusion
        projected = self.fc_proj(concatenated)

        # Residual block / Skip connection
        modality_embedding = projected + shared_feature
        
        return modality_embedding


class MissingModalityFeatureGeneration(nn.Module):
    """
    Generate features for missing modalities by averaging existing shared features.
    """
    def __init__(self):
        super(MissingModalityFeatureGeneration, self).__init__()

    def forward(self, shared_features, missing_indices, ablated, ablated_shape):
        """
        Args:
            shared_features (list): List of shared features for each modality, can contain None for missing modalities.
        Returns:
            list: List of generated features for missing modalities.
        """

        if not ablated:
            # Calculate the mean along the stack dimension (0) to get the average feature
            mean_feature = torch.mean(torch.stack(shared_features), dim=0)

            generated_features = [mean_feature for _ in missing_indices]
        else: 
            # Noise features for ablated shared encoder or missing modality features
            generated_features = [torch.randn(ablated_shape) for _ in missing_indices]

        return generated_features


class Decoder(nn.Module):
    """
    Decoder to predict the final output.

    Input: concatenated features
    Output: prediction
    """
    def __init__(
        self,
        number_class,
        num_of_sensor_channels,
        num_modalities,
        num_available_modalities,
        filter_num,
        ablate_shared_encoder,
        ablate_missing_modality_features
    ):
        super(Decoder, self).__init__()  
        
        # Flatten all dimensions before FC-layer
        self.flatten = nn.Flatten()

        self.fc_layer = nn.Linear(2 * filter_num * num_of_sensor_channels * num_modalities, number_class)
        


    def forward(self, concatenated_features):
        output = self.flatten(concatenated_features)
        output = self.fc_layer(output)

        return output


class ShaSpec(nn.Module):
    """
    ShaSpec model for multimodal sensor data.

    Args:
        input_shape (tuple): Shape of the input tensor.
        num_modalities (int): Number of modalities.
        miss_rate (float): Miss rate for the missing modality.
        num_classes (int): Number of classes.
        activation (str): Activation function (ReLU or Tanh).
        shared_encoder_type (str): Type of shared encoder.
        ablate_shared_encoder (bool): Whether to ablate the shared encoder.
        ablate_missing_modality_features (bool): Whether to ablate the missing modality features.
        config (dict): Configuration dictionary.
    """
    def __init__(
        self,
        input_shape,
        num_modalities,
        miss_rate,
        num_classes,
        activation,
        shared_encoder_type,
        ablate_shared_encoder,
        ablate_missing_modality_features,
        config):
        super(ShaSpec, self).__init__()

        self.filter_num = config["filter_num"]
        self.filter_size = config["filter_size"]
        self.sa_div = config["sa_div"]
        self.ablate_shared_encoder = ablate_shared_encoder
        self.ablate_missing_modality_features = ablate_missing_modality_features
        self.num_of_sensor_channels = input_shape[3]

        print("=" * 16, " ShaSpec Model Configuration ", "=" * 16)
        print("Number of total modalities: ", num_modalities)
        print("Selected miss rate: ", miss_rate)
        num_of_missing_modalities = ceil(num_modalities * miss_rate)
        self.num_available_modalities = num_modalities - num_of_missing_modalities
        print("Number of available modalities: ", self.num_available_modalities)
        print("Ablate shared encoder: ", self.ablate_shared_encoder)
        print("Ablate missing modality features: ", self.ablate_missing_modality_features)

        # Individual specific encoders
        self.specific_encoders = nn.ModuleList([SpecificEncoder(input_shape, self.filter_num, self.filter_size, activation, self.sa_div) for _ in range(self.num_available_modalities)])

        if not self.ablate_shared_encoder:
            # One shared encoder for all modalities
            self.shared_encoder = SharedEncoder(self.num_available_modalities, input_shape, self.filter_num, self.filter_size, activation, self.sa_div, shared_encoder_type)

        if not self.ablate_shared_encoder:
            # Residual block for fusion
            self.residual_block = ResidualBlock(self.filter_num)

        
        self.missing_modality_feature_generation = MissingModalityFeatureGeneration()

        self.decoder = Decoder(num_classes, self.num_of_sensor_channels, num_modalities, self.num_available_modalities, self.filter_num, ablate_shared_encoder, ablate_missing_modality_features)

    def forward(self, x_list, missing_indices):
        """
        x_list: List of tensors, each tensor represents a modality (complete and missing modalities included)
        missing_indices: List of indices for missing modalities
        """
        
        """ ================ Filter Out Missing Modalities ================"""
        available_indices = [modality for modality in range(len(x_list)) if modality not in missing_indices]
        available_modalities = [x_list[i] for i in available_indices]
        """ ================ Specific Encoders ================"""
        # Process with specific encoders for available modalities
        specific_features = [self.specific_encoders[i](available_modalities[i]) for i in range(len(available_modalities))]    

        shared_features = None
        """ ================ Shared Encoder ================"""
        if not self.ablate_shared_encoder:
            # Concatenate the modalities for shared processing
            concatenated_inputs = torch.cat(available_modalities, dim=3)  # Adjust dim according to how you concatenate
            shared_features = self.shared_encoder(concatenated_inputs)

        """ ================ Fuse Specific and Shared Features ================"""
        if not self.ablate_shared_encoder:
            # Fuse specific and shared features for available modalities
            fused_features = []
            for specific, shared in zip(specific_features, shared_features):
                fused_feature = self.residual_block(specific, shared)
                fused_features.append(fused_feature)
        else:
            fused_features = specific_features

        """ ================ Missing Modality Feature Generation ================"""
        ablated = self.ablate_shared_encoder or self.ablate_missing_modality_features
        ablated_shape = specific_features[0].shape
        generated_features = self.missing_modality_feature_generation(shared_features, missing_indices, ablated, ablated_shape)
        print("Generated features shape", generated_features[0].shape)

        # Insert the reconstructed modalities back at their original positions into fused_features
        for index, feature in sorted(zip(missing_indices, generated_features), key=lambda x: x[0]):
            fused_features.insert(index, feature)
        
        print("Fused features shape", fused_features[0].shape)

        """ ================ Decoder ================"""
        # Prepare for decoding
        concatenated_features = torch.cat(fused_features, dim=2)

        # Decode to get final predictions
        prediction = self.decoder(concatenated_features)

        return prediction
