import torch
import torch.nn as nn


class HARBox_CNN(nn.Module):
    def __init__(self, shrinkage_ratio=1):
        super(HARBox_CNN, self).__init__()
        self.shrinkage_ratio = shrinkage_ratio

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=int(16 * self.shrinkage_ratio), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(int(16 * self.shrinkage_ratio)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(16 * self.shrinkage_ratio),
                out_channels=int(32 * self.shrinkage_ratio),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(int(32 * self.shrinkage_ratio)),
        )

        self.linear = nn.Linear(int(32 * self.shrinkage_ratio) * 8 * 8, 5)

    def forward(self, x):
        self.batch_size = x.size(0)
        x = x.view(self.batch_size, 1, 30, 30)

        conv_output = torch.relu(self.conv1(x))
        conv_output = torch.relu(self.conv2(conv_output))

        # Dynamically calculate the flattened size instead of hardcoding it
        conv_output = conv_output.view(self.batch_size, -1)
        output = self.linear(conv_output)

        return output
