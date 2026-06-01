import torch
import torch.nn as nn

class PoseCNN(nn.Module):
    def __init__(self, num_input_frames=2):
        super(PoseCNN, self).__init__()
        self.num_input_frames = num_input_frames

        self.features = nn.Sequential(
            nn.Conv2d(3 * num_input_frames, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), kernel_size=1)

    def forward(self, x):
        out = self.features(x)
        out = self.pose_conv(out)

        out = out.mean(dim=(2, 3))

        out = 0.01 * out.view(-1, 6) 

        axisangle = out[:, :3]
        translation = out[:, 3:]

        return axisangle, translation