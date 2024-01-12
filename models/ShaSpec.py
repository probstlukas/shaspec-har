import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Specific Encoder for each modality


"""
class SpecificEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, domain_classes):
        super(SpecificEncoder, self).__init__()
        # Define a simple feedforward network as an example
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )
        # Domain classifier head
        self.domain_classifier = nn.Linear(feature_dim, domain_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        domain_logits = self.domain_classifier(features)
        return features, domain_logits

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


