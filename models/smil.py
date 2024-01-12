import torch
import torch.nn as nn
import torch.nn.functional as F

class SMIL(nn.Module):
    def __init__(self, num_classes, input_length, num_sensor_channels=1):
        super(SMIL, self).__init__()
        # Adjust these layers based on your specific HAR data
        self.conv1 = nn.Conv1d(in_channels=num_sensor_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Calculate the feature length after convolutional and pooling layers
        # This is a placeholder calculation and should be adjusted
        feature_length = self._calculate_feature_length(input_length)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * feature_length, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _calculate_feature_length(self, input_length):
        # Placeholder function for calculating feature length
        # Adjust this based on your network architecture and input size
        length_after_conv = input_length # Adjust based on convolutions
        length_after_pool = length_after_conv // 2 # Example for one pooling layer
        return length_after_pool

    def forward(self, x):
        # Forward pass through the network
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * self._calculate_feature_length(x.shape[2]))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage:
# num_classes = 10  # Example: Number of activities in HAR
# input_length = 100  # Example: Length of your time-series sensor data
# model = SMIL(num_classes, input_length)
# output = model(sensor_data)
