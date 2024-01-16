import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared and Specific Encoder Architektur gleich, aber Dimensionen anders, input shapes anders
# TODO: Paper noch mal genauer anschauen, gibt es Unterschiede zwischen Specific und Shared Encoder bzgl. Architektur?
class SpecificEncoder(nn.Module):
    """
    Specific Encoder for each modality.
    """
    def __init__(self, input_dim, feature_dim, domain_classes):
        super(SpecificEncoder, self).__init__()
        
        # 1. PART: Four convolutional layers for local context extraction
        # TODO: Pytorch docs lesen, stride auf 2 setzen. Wieso 2?
        self.conv1 = nn.Conv2d(input_shape[1], filter_num, (filter_size, 1), stride=2)
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=2)
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=2)
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1), stride=2)
        
        # 2. PART: Self-attention layer to allow all sensor channels to communicate with each other
        self.sa = SelfAttention(filter_num, sa_div)

        # 3. PART: FC layer for sensor channel fusion
        # Assuming the output of conv4 is [batch, filter_num, height, width]
        # and we want to combine the sensor channels (width dimension)
        conv_output_size = filter_num * input_shape[2] // (2**4)  # Adjust based on pooling and stride
        self.fc_fusion = nn.Linear(conv_output_size, domain_classes)
        
        # Domain classifier head
        # TODO: feature_dim instead of domain_classes as input vector?
        self.domain_classifier = nn.Linear(domain_classes, domain_classes)

    def forward(self, x):
        # TODO: reshape

        # Apply convolutional layers
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # Apply self-attention on each temporal dimension (along sensor and feature dimensions)
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        
    #     features = self.feature_extractor(x)
    #     domain_logits = self.domain_classifier(features)
    #     return features, domain_logits

class SharedEncoder(nn.Module):
    # Implement the shared encoder
    # ...
    pass

class ShaSpec(nn.Module):
    # def __init__(self, 
    #             input_shape, 
    #             nb_classes,
    #             filter_scaling_factor,
    #             config):
    def __init__(self, num_modalities, num_classes, input_dim, hidden_dim):
        super(ShaSpec, self).__init__()
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
