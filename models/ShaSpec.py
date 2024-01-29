import torch # For all things PyTorch
import torch.nn as nn # For torch.nn.Module, the parent object for PyTorch models
from models.Attend import SelfAttention # For the self-attention mechanism

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
        # Halving the length with each convolutional layer
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
            x (Tensor): The input tensor of shape (B=batch_size, F=filter_num, T=temp_length, C=num_of_sensor_channels).
        
        Returns:
            Tensor: The output tensor after applying convolutions, self-attention, and a fully connected layer.
        
        The function applies a series of convolutional layers to extract features from each sensor channel, followed by a self-attention mechanism to refine these features. 
        Finally, it reshapes and passes the output through a fully connected layer for sensor channel fusion.
        """
        print("-"*32)
        print("--> B x F x T x C")
        # B x F x T x C
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
        print("--> B x F' x T* x C")
        # --> B x F' x T^* x C

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
        # B x C x (T^* F')

        ### Pass through FC layer
        x = self.fc_fusion(x)
        print("After passing through fc layer: ", x.shape)
        # --> B x C x 2F'
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
        sa_div,
        weighted_sum = False
    ):
        super(SharedEncoder, self).__init__()

        self.weighted_sum = weighted_sum

        if not weighted_sum:
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
        # B F T C
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
        # Feature vector für jeden sensor channel, nicht time step!
        # Reshape the tensor to flatten/combine the last two dimensions
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # batch_size, num_of_sensor_channels, temp_length * filter_num
        # B C FT
        # Reshape and apply the FC layer
        x = self.fc_fusion(x)

        if not self.weighted_sum:
            print("Before splitting: ", x.shape)
            # Split the tensor back into N separate tensors along the second dimension C
            # Each tensor in the list corresponds to one modality
            split_tensors = torch.split(x, self.num_of_sensor_channels, dim=1)

            for i, tensor in enumerate(split_tensors):
                print(f"After splitting:{i} ", tensor.shape)
            return split_tensors

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
        if decoder_type == "fclayer":
            self.decoder = nn.Linear(2 * filter_num * modalities_num, number_class)
        elif decoder_type == "convtrans":
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(2 * filter_num * modalities_num, filter_num, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(filter_num, number_class, kernel_size=3, stride=2)
            )

    def forward(self, concatenated_features):
        # Decode the features based on the decoder type
        if self.decoder_type == "fclayer":
            output = self.decoder(concatenated_features.view(concatenated_features.size(0), -1))
        elif self.decoder_type == "convtrans":
            output = self.decoder(concatenated_features.view(concatenated_features.size(0), -1, 1, 1))
        return output


class ShaSpec(nn.Module):
    def __init__(
        self,
        input,
        number_class, 
        modalities_num,
        sa_div                         = 8,
        filter_num                     = 64,        
        filter_size                    = 5,
        activation = "ReLU",
        decoder_type = "fclayer", # or "convtrans"
        weighted_sum = False # Variant of SharedEncoder
    ):
        super(ShaSpec, self).__init__()

        self.weighted_sum = weighted_sum

        # Individual specific encoders
        self.specific_encoders = nn.ModuleList([SpecificEncoder(input, filter_num, filter_size, activation, sa_div) for _ in range(modalities_num)])

        # One shared encoder for all modalities
        self.shared_encoder = SharedEncoder(modalities_num, input, filter_num, filter_size, activation, sa_div)

        self.residual_block = ResidualBlock(filter_num)

        self.decoder = Decoder( number_class, modalities_num, filter_num, decoder_type)

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
        if self.weighted_sum:
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

