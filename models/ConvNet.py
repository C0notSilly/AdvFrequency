import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x, return_feat=False):
        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if return_feat:
            return out4, self.fc3(out4)
        else:
            return self.fc3(out4)

    def get_logits_feat(self, x):
        B = x.shape[0]
        output = self.mp(self.relu1(self.conv1(x)))
        output = self.mp(self.relu2(self.conv2(output)))

        feat = output.clone()

        output = output.view(B, -1)
        output = self.relu3(self.fc1(output))
        output = self.relu4(self.fc2(output))

        output = self.fc3(output)
        return output, feat
