import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np

class BaseDeepConv(nn.Module):
    def __init__(self, input_chann=3):
        super(BaseDeepConv, self).__init__()
        # /2
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False).cuda()
        self.bn0 = nn.BatchNorm2d(64).cuda()
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).cuda()

        # /2
        self.conv1 = nn.Conv2d(64, 128, 3, padding=1).cuda()
        self.bn1 = nn.BatchNorm2d(128).cuda()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1).cuda()
        self.bn2 = nn.BatchNorm2d(128).cuda()
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1).cuda()
        self.bn3 = nn.BatchNorm2d(128).cuda()
        self.conv4 = nn.Conv2d(128, 128, 1, stride=2, bias=False).cuda()
        self.bn4 = nn.BatchNorm2d(128).cuda()

        # /3
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1).cuda()
        self.bn5 = nn.BatchNorm2d(256).cuda()
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1).cuda()
        self.bn6 = nn.BatchNorm2d(256).cuda()
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1).cuda()
        self.bn7 = nn.BatchNorm2d(256).cuda()
        self.conv8 = nn.Conv2d(256, 256, 1, stride=2, bias=False).cuda()
        self.bn8 = nn.BatchNorm2d(256).cuda()

        # /4
        self.conv9 = nn.Conv2d(256, 512, 3, padding=1).cuda()
        self.bn9 = nn.BatchNorm2d(512).cuda()
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1).cuda()
        self.bn10 = nn.BatchNorm2d(512).cuda()
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1).cuda()
        self.bn11 = nn.BatchNorm2d(512).cuda()
        self.conv12 = nn.Conv2d(512, 512, 1, stride=2, bias=False).cuda()
        self.bn12 = nn.BatchNorm2d(512).cuda()

        self.relu = nn.ReLU(inplace=True).cuda()  # 将输出覆盖到输入中

    def forward(self, x):
        x = self.pool0(self.relu(self.bn0(self.conv0(x))))

        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(x1))))))
        x3 = self.relu(self.bn4(self.conv4(x1 + x2)))

        x4 = self.relu(self.bn5(self.conv5(x3)))
        x5 = self.relu(self.bn7(self.conv7(self.relu(self.bn6(self.conv6(x4))))))
        x6 = self.relu(self.bn8(self.conv8(x4 + x5)))

        x7 = self.relu(self.bn9(self.conv9(x6)))
        x8 = self.relu(self.bn11(self.conv11(self.relu(self.bn10(self.conv10(x7))))))
        x9 = self.relu(self.bn12(self.conv12(x7 + x8)))

        return x9



class JointRegression(nn.Module):
    ''' joint angle regression from hand embedding space'''

    def __init__(self, input_size=128, output_size=24):
        super(JointRegression, self).__init__()

        hidden_size = input_size // 2
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.reg = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        angle = self.reg(x)

        return angle


class NewJointRegression(nn.Module):
    def __init__(self, input_size=128, output_size=24):
        super(NewJointRegression, self).__init__()
        self.reg = nn.Linear(input_size, output_size).cuda()

    def forward(self, x):
        angle = self.reg(x)

        return angle


class Discriminator_Embedding(nn.Module):
    ''' hand embedding space discriminator'''

    def __init__(self, input_size=128):
        super(Discriminator_Embedding, self).__init__()

        hidden_size = input_size // 2
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        output = F.sigmoid(self.fc3(x))

        return output


class MoRe(nn.Module):
    def __init__(self, input_size=100, embedding_size=128, joint_size=22):
        super(NewTeachingTeleModel, self).__init__()
        self.hand = BaseDeepConv(input_chann=3)
        self.feature_size = 512 * 16
        self.embedding_size = embedding_size
        self.joint_size = joint_size

        self.encoder_hand = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size * 4),
            nn.BatchNorm1d(self.embedding_size * 4),
            nn.ReLU(),
            nn.Linear(self.embedding_size * 4, self.embedding_size * 2),
            nn.BatchNorm1d(self.embedding_size * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        ).cuda()

        self.hand_reg = NewJointRegression(input_size=self.embedding_size, output_size=self.joint_size)

    def forward(self, x, is_human=True):
        x = self.hand(x).view(-1, self.feature_size)
        embedding = self.encoder_hand(x)
        joint = self.hand_reg(embedding)


        return embedding, joint



if __name__ == '__main__':
    x = torch.ones(3, 100, 100)

    t = MoRe(100, 128, 24)
    a, b = t.forward(x, True)
    print(b.shape)
    print(a.shape)
