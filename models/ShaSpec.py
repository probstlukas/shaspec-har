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
        # Halving the length with each convolutional layer
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
        # Create a dummy input tensor based on the input shape
        x = torch.randn(input_shape)

        # Pass the dummy input through the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Return temp_length, i.e. now downsampled_length
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
            # Split the tensor back into separate tensors for each modality
            split_tensors = torch.split(x, self.num_of_sensor_channels, dim=1)
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

    def forward(self, specific_features, shared_features):
        # List of extracted features
        modality_embeddings = []
        for specific, shared in zip(specific_features, shared_features):
            # Concatenate
            concatenated = torch.cat((specific, shared), dim=2)
            print("CONCATENATED: ", concatenated.shape)

            # Linear projection for fusion
            projected = self.fc_proj(concatenated)
            print("PROJECTED: ", projected.shape)
            print("-"*32)
            print(projected.shape)
            print(shared.shape)

            # Residual block / Skip connection
            modality_embedding = projected + shared
            print("MODALITY EMBEDDING: ", modality_embedding.shape)
            modality_embeddings.append(modality_embedding)

class Decoder(nn.Module):
    def __init__(
        self,
        number_class,
        modalities_num,
        filter_num,
        decoder_type
    ):
        super(Decoder, self).__init__()

        # Decoder
        # TODO: Check input/output shape
        if decoder_type == "FC":
            self.decoder = nn.Linear(2 * filter_num * modalities_num, number_class)
        elif decoder_type == "ConvTrans":
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(2 * filter_num * modalities_num, filter_num, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(filter_num, number_class, kernel_size=3, stride=2)
            )

    def forward(self, concatenated_features):
        # Decode the features based on the decoder type
        if self.decoder_type == "FC":
            output = self.decoder(concatenated_features.view(concatenated_features.size(0), -1))
        elif self.decoder_type == "ConvTrans":
            output = self.decoder(concatenated_features.view(concatenated_features.size(0), -1, 1, 1))
        return output


class ShaSpec(nn.Module):
    def __init__(
        self,
        input,
        number_class, 
        filter_num,
        sa_div,
        modalities_num = 6,# TODO
        filter_size         = 5,
        activation          = "ReLU",
        decoder_type        = "FC", # FC ConvTrans
        shared_encoder_type = "concatenated" # concatenated weighted
    ):
        super(ShaSpec, self).__init__()

        self.shared_encoder_type = shared_encoder_type

        # Individual specific encoders
        self.specific_encoders = nn.ModuleList([SpecificEncoder(input, filter_num, filter_size, activation, sa_div) for _ in range(modalities_num)])

        # One shared encoder for all modalities
        self.shared_encoder = SharedEncoder(modalities_num, input, filter_num, filter_size, activation, sa_div, shared_encoder_type)

        self.residual_block = ResidualBlock(filter_num)

        self.decoder = Decoder(number_class, modalities_num, filter_num, decoder_type)

    def forward(self, x_list):
        """
        x_list: list of input tensors, each corresponding to one modality.
        """
        print("-"*16, "Specific Encoder", "-"*16)
        # List of all specific features
        specific_features = [encoder(x) for encoder, x in zip(self.specific_encoders, x_list)]
        print("SPECIFIC FEATURES: ", specific_features)

        print("-"*16, "Shared Encoder", "-"*16)
        # Depending on chosen SharedEncoder approach
        if self.shared_encoder_type == "concatenated":
            shared_features = [encoder(x) for encoder, x in zip(self.shared_features, x_list)]
        else:
            shared_features = self.shared_encoder(torch.cat(x_list, dim=3))
        print("SHARED FEATURES: ", shared_features)
        # Split the shared features for each modality

        modality_embeddings = self.residual_block(specific_features, shared_features)

        # Prepare features for decoder by concatenating them along the F dimension
        concatenated_features = torch.cat(modality_embeddings, dim=2)
        print("Shape of concatenated features: ", concatenated_features.shape)

        prediction = self.decoder(concatenated_features)

        return prediction
