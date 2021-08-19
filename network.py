import torch
import torch.nn as nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

        # Net1: 15features * 2sensors
        self.fc1 = nn.Conv1d(1, 16, kernel_size=5, stride=5)  #
        self.fc2 = nn.Conv1d(16, 64, kernel_size=3, stride=3)  # 2*64

        # Net2
        # self.block_1 = nn.Sequential(
        #     # Layer 1
        #     nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     # Layer 2
        #     nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     # Layer 3
        #     nn.Conv1d(16, 16, kernel_size=1, stride=1),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.MaxPool1d(stride=2, kernel_size=2)
        # )
        # self.block_2 = nn.Sequential(
        #     # Layer 1
        #     nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     # Layer 2
        #     nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     # Layer 3
        #     nn.Conv1d(64, 64, kernel_size=1, stride=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(stride=2, kernel_size=2)
        # )
        # self.fc3 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        #

        # # Net 3: 10 features * 1 sensors
        # self.fc1 = nn.Conv1d(1, 16, kernel_size=5, stride=5)  #
        # self.fc2 = nn.Conv1d(16, 64, kernel_size=2, stride=1)  # 1*64
        #
        # # Net 3: gyr 15 + lacc 10
        # self.fc1 = nn.Conv1d(1, 16, kernel_size=5, stride=5)  #
        # self.fc2 = nn.Conv1d(16, 64, kernel_size=3, stride=1)  # 3*64

        # Net4: 15features * 3sensors
        self.fc1 = nn.Conv1d(1, 16, kernel_size=5, stride=5)  #
        self.fc2 = nn.Conv1d(16, 64, kernel_size=3, stride=3)  # 3*64

        # # Net for frequency features
        # self.fc1 = nn.Conv1d(1, 16, kernel_size=3, stride=3)  #
        # self.fc2 = nn.Sequential(
        #     nn.Conv1d(16, 64, kernel_size=3, stride=3),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 128, kernel_size=3, stride=3))

        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 1024),
            # nn.Linear(64 * 4, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, output_size)  # 之前validation的是48%
        )

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        # out = self.fc3(out)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.classifier(out)
        out = self.tanh(out)
        out = F.softmax(out, dim=1)

        return out


# 参考GoogleNet网络，输入六个特征图，然后每个做个变换，再合并，合并后做线性变换
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = []
