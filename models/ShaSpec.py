import torch # For all things PyTorch
import torch.nn as nn # For torch.nn.Module, the parent object for PyTorch models
from models.Attend import SelfAttention # For the self-attention mechanism

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
            nn.Conv2d(input_shape[1], filter_num, (filter_size, 1), stride=(2, 1)),
            self.get_activation(activation),
            nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=(2, 1)),
            self.get_activation(activation),
            nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=(2, 1)),
            self.get_activation(activation),
            nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=(2, 1)),
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
        # print("Input shape: ", x.shape)
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

        # print("After conv block: ", x.shape)

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

        # print("Shared encoder input shape in SharedEncoder: ", input_shape) # 64 1 125 36
        self.be = BaseEncoder(
            input_shape,
            filter_num,
            filter_size, 
            activation, 
            sa_div
        )
        # print("Shared encoder input shape: ", input_shape)
        
    def forward(self, x):
        x = self.be(x)       
        # print("Shared feature shape in forward func: ", x.shape)
        shared_features = torch.tensor_split(x, self.num_available_modalities, dim=1)
        # print("Each shared feature shape: ", shared_features[0].shape)
        return shared_features


class ResidualBlock(nn.Module):
    def __init__(
        self,
        filter_num,
    ):
        super(ResidualBlock, self).__init__()

        # Linear projection
        self.fc_proj = nn.Linear(4 * filter_num, 2 * filter_num)

    def forward(self, specific_feature, shared_feature):
        # Concatenate
        # print("Specific feature shape: ", specific_feature.shape)
        # print("Shared feature shape: ", shared_feature.shape)
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

    def forward(self, shared_features, missing_indices):
        """
        Args:
            shared_features (list): List of shared features for each modality, can contain None for missing modalities.
        Returns:
            list: List of generated features for missing modalities.
        """

        # Calculate the mean along the stack dimension (0) to get the average feature
        mean_feature = torch.mean(torch.stack(shared_features), dim=0)
        # print("Mean feature: ", mean_feature)

        generated_features = [mean_feature for _ in missing_indices]
        # print("Missing indices length for gen. features: ", len(missing_indices))
        # print("Generated features length: ", generated_features)
        # print("Generated features: ", generated_features)

        return generated_features


class Decoder(nn.Module):
    def __init__(
        self,
        number_class,
        num_of_sensor_channels,
        num_modalities,
        filter_num,
        decoder_type
    ):
        super(Decoder, self).__init__()
        self.decoder_type = decoder_type
        # Decoder
        if decoder_type == "FC":
            # Flatten all dimensions before FC-layer
            self.flatten = nn.Flatten()
            # 42 * 5 * 2
            # C x 2F' = 42 * 5 * 2 = 420
            # print("Filter num: ", filter_num)
            # print("Num of sensor channels: ", num_of_sensor_channels)
            # print("modalities num: ", num_modalities)
            self.fc_layer = nn.Linear(2 * filter_num * num_of_sensor_channels * num_modalities, number_class)

    def forward(self, concatenated_features):
        output = self.flatten(concatenated_features)
        # print("Shape after flattening: ", output.shape)
        output = self.fc_layer(output)
        # print("Shape after fc layer: ", output.shape)

        return output


class ShaSpec(nn.Module):
    def __init__(
        self,
        input,
        num_modalities,
        miss_rate,
        num_classes,
        filter_num,
        filter_size,
        sa_div,
        activation          = "ReLU", # ReLU Tanh
        decoder_type        = "FC", # FC ConvTrans
        shared_encoder_type = "concatenated" # concatenated weighted
    ):
        super(ShaSpec, self).__init__()

        # print("ShaSpec input shape: ", input)
        # print("miss rate: ", miss_rate)
        print("Filter num: ", filter_num)
        self.num_of_sensor_channels = input[3]
        self.shared_encoder_type = shared_encoder_type

        # print("==== ShaSpec Model ====")
        # print("Num of modalities: ", num_modalities)
        # print("Miss rate: ", miss_rate)
        self.num_available_modalities = int(num_modalities * (1 - miss_rate))
        # print("Num of available modalities: ", self.num_available_modalities)

        # Individual specific encoders
        self.specific_encoders = nn.ModuleList([SpecificEncoder(input, filter_num, filter_size, activation, sa_div) for _ in range(self.num_available_modalities)])

        # One shared encoder for all modalities
        self.shared_encoder = SharedEncoder(self.num_available_modalities, input, filter_num, filter_size, activation, sa_div, shared_encoder_type)

        self.residual_block = ResidualBlock(filter_num)

        self.missing_modality_feature_generation = MissingModalityFeatureGeneration()

        self.decoder = Decoder(num_classes, self.num_of_sensor_channels, num_modalities, filter_num, decoder_type)

    def forward(self, x_list, missing_indices):
        """
        x_list: List of tensors, each tensor represents a modality
        missing_indices: List of indices of missing modalities
        """
        # print("Available modalities: ", self.num_available_modalities)
        # print("Input list: ", x_list)
        # print("Missing index: ", missing_indices)

        # Filter out missing modalities
        available_indices = [modality for modality in range(len(x_list)) if modality not in missing_indices]
        # print("Available indices: ", available_indices)
        available_modalities = [x_list[i] for i in available_indices]
        # print("Available x list length: ", len(available_modalities))

        # Process with specific encoders for available modalities
        specific_features = [self.specific_encoders[i](available_modalities[i]) for i in range(len(available_modalities))]    
        # print("Specific features length: ", len(specific_features))



        # print("Available modality shape: ", available_modalities[0].shape)
        # Concatenate the modalities for shared processing
        concatenated_inputs = torch.cat(available_modalities, dim=3)  # Adjust dim according to how you concatenate
        shared_features = self.shared_encoder(concatenated_inputs)
        # print("Shared features shape: ", shared_features[0].shape)
    

        # Fuse specific and shared features for available modalities
        fused_features = []
        for specific, shared in zip(specific_features, shared_features):
            fused_feature = self.residual_block(specific, shared)
            fused_features.append(fused_feature)

        # Reconstruct missing modalities using shared features
        generated_features = self.missing_modality_feature_generation(shared_features, missing_indices)

        # Insert the reconstructed modalities back at their original positions into fused_features
        for index, feature in sorted(zip(missing_indices, generated_features), key=lambda x: x[0]):
            fused_features.insert(index, feature)

        # Prepare for decoding
        concatenated_features = torch.cat(fused_features, dim=2)

        # Decode to get final predictions
        prediction = self.decoder(concatenated_features)

        return prediction