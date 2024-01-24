import torch
import torch.nn as nn
from models.Attend import SelfAttention

class SpecificEncoder(nn.Module):
    """
    Specific Encoder to capture the unique characteristics of each sensor channel.

    1 modality as input, 1 specific feature as output
    """
    def __init__(
        self,
        input_shape,
        filter_num,
        filter_size,
        activation,
        sa_div
    ):
        super(SpecificEncoder, self).__init__()

        ### Part 1: Convolutional layers for local context extraction
        # Halving the length by 2 with each convolutional layer (assuming height is the temporal dimension)
        self.conv1 = nn.Conv2d(input_shape[1], filter_num, (filter_size, 1), stride=(2, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=(2, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=(2, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=(2, 1))
        
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        ### Part 2: Self-attention layer for inter-sensor channel communication
        self.sa = SelfAttention(filter_num, sa_div)

        ### Part 3: FC layer for sensor channel fusion
        # Sequence length after applying conv layers
        seq_length = self.get_shape_of_conv4(input_shape)[2]
        # Output feature dimension
        # Multiply To reduce feature space
        feature_dim = 2 * filter_num
        self.fc_fusion = nn.Linear(filter_num * seq_length, feature_dim)

    def forward(self, x):
        """
        Perform a forward pass of the SpecificEncoder model.

        Args:
            x (Tensor): The input tensor of shape (batch_size, filter_num, sequence_length, num_of_sensor_channels).
        
        Returns:
            Tensor: The output tensor after applying convolutions, self-attention, and a fully connected layer.
        
        The function applies a series of convolutional layers to extract features from each sensor channel, followed by a self-attention mechanism to refine these features. 
        Finally, it reshapes and passes the output through a fully connected layer for sensor channel fusion.
        """

        print("--> batch_size, F=1, T, C")
        print("Before applying conv layers: ", x.shape)
        ### Apply convolutional layers
        x = self.activation(self.conv1(x))
        print("After applying conv layer 1: ", x.shape)
        x = self.activation(self.conv2(x))
        print("After applying conv layer 2: ", x.shape)
        x = self.activation(self.conv3(x))
        print("After applying conv layer 3: ", x.shape)
        x = self.activation(self.conv4(x))
        print("After applying conv layers: ", x.shape)
        print("--> batch_size, F=filter_num, T*, C")

        ### Apply self-attention to each time step in the sequence
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
        
        print("After applying self-attention: ", refined.shape)

        ### Reshaping and reordering the output for the FC layer
        # Swap filter_num with seq_length
        x = refined.permute(0, 2, 1, 3)
        print("After permuting: ", x.shape)
        # batch_size, num_of_sensor_channels, filter_num, temporal length
        # Reshape the tensor to flatten/combine the last two dimensions
        x = x.reshape(x.shape[0], x.shape[1], -1)
        print("After reshaping: ", x.shape)
        # batch_size, num_of_sensor_channels, temp_length * filter_num
        # temp_length too big --> fc layer

        ### Pass through FC layer
        x = self.fc_fusion(x)
        print("After passing through fc layer: ", x.shape)

        return x
    
    def get_shape_of_conv4(self, input_shape):
        """
        Compute the shape of the output after the fourth convolutional layer.
        
        Args:
        - input_shape (tuple): The shape of the input tensor.
        - filter_num (int): Number of filters in the convolutional layers.
        - filter_size (int): Size of the kernel in the convolutional layers.

        Returns:
        - tuple: The shape of the output after the fourth convolutional layer.
        """
        # Create a dummy input tensor based on the input shape
        x = torch.randn(input_shape)

        # Pass the dummy input through the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Return the shape of the output
        return x.shape


class SharedEncoder(nn.Module):
    """
    Shared Encoder to to extract common features that are informative accross all sensor channels.

    N modalities as input, N shared features as output
    """
    def __init__(
        self,
        modalities_num,
        input_shape,
        filter_num,
        filter_size,
        activation,
        sa_div
    ):
        super(SharedEncoder, self).__init__()

        self.modalities_num = modalities_num
        self.num_of_sensor_channels = input_shape[3]

        # Concatenate modalities
        input_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] * modalities_num)
        print("Sha Enc input shape: ", input_shape)

        ### Part 1: Convolutional layers for local context extraction
        # Halving the length by 2 with each convolutional layer (assuming height is the temporal dimension)
        self.conv1 = nn.Conv2d(input_shape[1], filter_num, (filter_size, 1), stride=(2, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=(2, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=(2, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=(2, 1))
        
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        ### Part 2: Self-attention layer for inter-sensor channel communication
        self.sa = SelfAttention(filter_num, sa_div)

        ### Part 3: FC layer for sensor channel fusion
        # Sequence length after applying conv layers
        seq_length = self.get_shape_of_conv4(input_shape)[2]
        # Output feature dimension
        # Multiply To reduce feature space
        feature_dim = 2 * filter_num
        self.fc_fusion = nn.Linear(filter_num * seq_length, feature_dim)

    def forward(self, x):
        """
        Perform a forward pass of the SharedEncoder model.

        Args:
            x (Tensor): The input tensor of shape (batch_size, filter_num, sequence_length, num_of_sensor_channels).
        
        Returns:
            Tensor: The output tensor after applying convolutions, self-attention, and a fully connected layer.
        
        The function applies a series of convolutional layers to extract features from each sensor channel, followed by a self-attention mechanism to refine these features. 
        Finally, it reshapes and passes the output through a fully connected layer for sensor channel fusion.
        """
        # B F T C
        # @ Yexu: L = T? Sequence length = temp length?
        print(x.shape)
        ### Apply convolutional layers
        x = self.activation(self.conv1(x))
        print(x.shape)
        x = self.activation(self.conv2(x))
        print(x.shape)
        x = self.activation(self.conv3(x))
        print(x.shape)
        x = self.activation(self.conv4(x))
        print(x.shape)

        ### Apply self-attention to each time step in the sequence
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
        
        print("After refined:", refined.shape)


        ### Reshaping and reordering the output for the FC layer
        # Before: (batch_size, filter_num, sequence_length, num_of_sensor_channels).
        # Swap filter_num with seq_length
        x = refined.permute(0, 2, 1, 3)
        # @ Yexu: Don't we want to flatten Num_of_sensor_channels and Filter_num?
        # This gives us temp_length * filter_num, but don't we want temp_length * num_of_sensor_channels?
        # x = refined.permute(0, 3, 1, 2)
        # batch_size, num_of_sensor_channels, filter_num, temporal length

        # Reshape the tensor to flatten/combine the last two dimensions
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # batch_size, num_of_sensor_channels, temp_length * filter_num

        # Reshape and apply the FC layer
        x = self.fc_fusion(x)

        # Reshape the output to separate the modalities
        # TODO: Each modality should have an output shape of batch_size, num_of_sensor_channels, temp_length * filter_num
        #x = x.reshape(x.shape[0], -1, x.shape[2], self.modalities_num)
        print("Before splitting: ", x.shape)
        # Split the tensor back into N separate tensors along the second dimension C
        # Each tensor in the list corresponds to one modality
        split_tensors = torch.split(x, self.num_of_sensor_channels, dim=1)
        
        return split_tensors
    
    def get_shape_of_conv4(self, input_shape):
        """
        Compute the shape of the output after the fourth convolutional layer.
        
        Args:
        - input_shape (tuple): The shape of the input tensor.
        - filter_num (int): Number of filters in the convolutional layers.
        - filter_size (int): Size of the kernel in the convolutional layers.

        Returns:
        - tuple: The shape of the output after the fourth convolutional layer.
        """
        # Create a dummy input tensor based on the input shape
        x = torch.randn(input_shape)

        # Pass the dummy input through the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Return the shape of the output
        return x.shape


class ShaSpec(nn.Module):
    def __init__(
            self, 
            num_modalities, 
            num_classes, 
            input_dim, 
            hidden_dim,
            filter_scaling_factor=1,
            config = None):
        super(ShaSpec, self).__init__()

        # Scale the hidden layer dimensions as specified by the configuration, allowing for 
        # flexible model capacity adjustments via the filter_scaling_factor multiplier.
        self.hidden_dim = int(filter_scaling_factor * config["hidden_dim"]);
        self.num_modalities = num_modalities
        self.specific_encoders = nn.ModuleList([SpecificEncoder(input_dim, hidden_dim) for _ in range(num_modalities)])
        self.shared_encoder = SharedEncoder(input_dim, hidden_dim)
        self.fc_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_list):
        # x_list is a list of modality data. Missing modalities are represented by None
        specific_features = [self.specific_encoders[i](x) if x is not None else None for i, x in enumerate(x_list)]
        shared_features = self.shared_encoder(torch.cat([x for x in x_list if x is not None], dim=0))

        # Handling missing modalities
        # Replace None with generated features using the shared encoder
        specific_features = [sf if sf is not None else shared_features for sf in specific_features]

        # Combine specific and shared features
        combined_features = [self.fc_proj(torch.cat((sf, shared_features), dim=1)) for sf in specific_features]

        # Decoder to make the final prediction
        # Assuming we're concatenating the features from different modalities

        # !!! the decoder in Fig. 1 and Fig. 2 only works for segmentation. 
        # For classification, the fused features are fed into fully connected (FC) layers.
        combined_features = torch.cat(combined_features, dim=1)
        output = self.decoder(combined_features)

        return output
