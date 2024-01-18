import torch
import torch.nn as nn
from models.Attend import SelfAttention

class SpecificEncoder(nn.Module):
    """
    Specific Encoder to capture the unique characteristics of each sensor channel.
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
        print("A")
        ### Part 1: Convolutional layers for local context extraction
        # Halving the length by 2 with each convolutional layer (assuming height is the temporal dimension)
        self.conv1 = nn.Conv2d(input_shape[1], filter_num, (filter_size, 1), stride=2)
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=2)
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=2)
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=2)
        
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        ### Part 2: Self-attention layer for inter-sensor channel communication
        self.sa = SelfAttention(filter_num, sa_div)

        ### Part 3: Fully connected layer for sensor channel fusion
        # Calculate the size for fully connected layer input
        num_sensor_channels = input_shape[3]  # Number of sensor channels
        conv_output_size = filter_num * num_sensor_channels
        feature_dim = 2 * filter_num  # Output feature dimension
        self.fc_fusion = nn.Linear(conv_output_size, feature_dim)

    def forward(self, x):

        # Apply convolutional layers
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # Apply self-attention
        # The input and output shapes remain the same [batch_size, num_features, sequence_length]
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )

        # Reshape for the fully connected layer
        x = refined.permute(3, 0, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        # Pass through fully connected layer
        x = self.fc_fusion(x)
        
        return x

class SharedEncoder(nn.Module):
    """
    Shared Encoder to to extract common features that are informative accross all sensor channels.
    """
    # Implement the shared encoder
    # Similar to SpecificEncoder, but different input shape
    pass

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
