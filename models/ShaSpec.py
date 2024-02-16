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


class SpecificEncoder(BaseEncoder):
    """
    Specific Encoder to capture the unique characteristics of each modality.

    Input: modality
    Output: specific feature
    """
    def __init__(self, input_shape, filter_num, filter_size, activation, sa_div):
        # Call the constructor of BaseEncoder
        super(SpecificEncoder, self).__init__(input_shape, filter_num, filter_size, activation, sa_div)

    def forward(self, x):
        # Call the forward method of BaseEncoder
        x = super(SpecificEncoder, self).forward(x)
        
        return x


class SharedEncoder(BaseEncoder):
    """
    Shared Encoder to to extract common features that are informative accross all sensor channels.

    Input: N modalities
    Output: N shared features as output
    """
    def __init__(self, modalities_num, input_shape, filter_num, filter_size, activation, sa_div, shared_encoder_type):
        self.modalities_num = modalities_num
        if shared_encoder_type == "concatenated":
            # Concatenate modalities before initializing the base class
            input_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] * modalities_num)
        self.num_of_sensor_channels = input_shape[3]
        
        # Now that input_shape is correctly set, we can initialize the base class
        super(SharedEncoder, self).__init__(input_shape, filter_num, filter_size, activation, sa_div)

        self.shared_encoder_type = shared_encoder_type

    def forward(self, x):
        # Call forward method of the base class
        x = super(SharedEncoder, self).forward(x)

        if self.shared_encoder_type == "concatenated":
            split_tensors = torch.tensor_split(x, self.modalities_num, dim=1)
            return split_tensors
                
        # If not concatenated, just return x
        return x

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
        concatenated = torch.cat((specific_feature, shared_feature), dim=2)
        # print("CONCATENATED: ", concatenated.shape)

        # Linear projection for fusion
        projected = self.fc_proj(concatenated)
        # print("PROJECTED: ", projected.shape)
        # print("-"*32)
        # print(projected.shape)
        # print(shared_feature.shape)

        # Residual block / Skip connection
        modality_embedding = projected + shared_feature
        # print("MODALITY EMBEDDING: ", modality_embedding.shape)

        return modality_embedding


class MissingModalityFeatureGeneration(nn.Module):
    """
    Generate features for missing modalities by averaging existing shared features.
    """
    def __init__(self):
        super(MissingModalityFeatureGeneration, self).__init__()

    def forward(self, shared_features):
        """
        Args:
            shared_features (list): List of shared features for each modality, can contain None for missing modalities.
        Returns:
            list: List of shared features with generated features replacing None for missing modalities.
        """

        # Filter out None values and stack existing shared features for averaging
        valid_features = torch.stack([feature for feature in shared_features if feature is not None], dim=0)
        
        # Calculate the mean along the stack dimension (0) to get the average feature
        average_feature = torch.mean(valid_features, dim=0, keepdim=True)
        
        # Replace None in shared_features with the average feature
        generated_features = [feature if feature is not None else average_feature for feature in shared_features]
        
        return generated_features


class Decoder(nn.Module):
    def __init__(
        self,
        number_class,
        num_of_sensor_channels,
        modalities_num,
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
            # print("modalities num: ", modalities_num)
            self.fc_layer = nn.Linear(2 * filter_num * num_of_sensor_channels * modalities_num, number_class)
            
        #elif decoder_type == "ConvTrans":
            # @TODO: Ask Yexu what his idea was for this decoder
            # Check input/output shape. ConvTranspose2d for upsampling and then Conv2d for downsampling to get the final output?
            # self.decoder = nn.Sequential(
            #     nn.ConvTranspose2d(2 * filter_num * modalities_num, filter_num, kernel_size=3, stride=2),
            #     nn.ReLU(),
            #     nn.ConvTranspose2d(filter_num, number_class, kernel_size=3, stride=2)
            # )

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
        modalities_num,
        # list with tensors, pos1: pytorch array, pos2: pytorch NaN tensor, ...
        classes_num,
        filter_num,
        filter_size,
        sa_div,
        activation          = "ReLU", # ReLU Tanh
        decoder_type        = "FC", # FC ConvTrans
        shared_encoder_type = "concatenated" # concatenated weighted
    ):
        super(ShaSpec, self).__init__()

        self.num_of_sensor_channels = input[3]
        self.shared_encoder_type = shared_encoder_type

        # Individual specific encoders
        self.specific_encoders = nn.ModuleList([SpecificEncoder(input, filter_num, filter_size, activation, sa_div) for _ in range(modalities_num)])

        # One shared encoder for all modalities
        self.shared_encoder = SharedEncoder(modalities_num, input, filter_num, filter_size, activation, sa_div, shared_encoder_type)

        self.residual_block = ResidualBlock(filter_num)

        self.missing_modality_feature_generation = MissingModalityFeatureGeneration()

        self.decoder = Decoder(classes_num, self.num_of_sensor_channels, modalities_num, filter_num, decoder_type)

    def forward(self, x_list):
        """
        x_list: list of input tensors, each corresponding to one modality.
        """
        # for x in x_list:
        #     print("Input shape: ", x.shape)

        # print("-"*16, "Specific Encoder", "-"*16)
        # List of all specific features
        specific_features = [encoder(x) for encoder, x in zip(self.specific_encoders, x_list)]
        # for feature in specific_features:
        #     print("SPECIFIC FEATURE SHAPE: ", feature.shape)
        # print("SPECIFIC FEATURES: ", specific_features)
        # print("SPECIFIC FEATURES LENGTH: ", len(specific_features))

        # print("-"*16, "Shared Encoder", "-"*16)
        # Process inputs through the shared encoder based on the chosen type
        if self.shared_encoder_type == "concatenated":
            # Concatenate the modalities for shared processing
            concatenated_inputs = torch.cat(x_list, dim=3)  # Adjust dim according to how you concatenate
            shared_features = self.shared_encoder(concatenated_inputs)
        # elif self.shared_encoder_type == "weighted":

        #     shared_features = []
        #     for x in x_list:
        #         shared_features.append(self.share_encoder(x))
        # print("Shared features: ", shared_features)
        # for feature in shared_features:
        #     print("SHARED FEATURE SHAPE: ", feature.shape)
        # print("SHARED FEATURES LENGTH: ", len(shared_features))

        # Generate features for missing modalities
        #shared_features = self.missing_modality_feature_generation(shared_features)

        fused_features = []
        for specific, shared in zip(specific_features, shared_features):
            # Process each pair through the residual block
            fused_feature = self.residual_block(specific, shared)
            fused_features.append(fused_feature)
        # print("MODALITY EMBEDDINGS: ", fused_features)

        # Prepare features for decoder by concatenating them along the F dimension
        concatenated_features = torch.cat(fused_features, dim=2)
        # print("Shape of concatenated features: ", concatenated_features.shape)

        prediction = self.decoder(concatenated_features)

        # print("Prediction shape: ", prediction.shape)

        return prediction
