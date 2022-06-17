import torch
import torch.nn as nn
from torchvision import models


class LNet(nn.Module):
    def __init__(self):
        super(LNet, self).__init__()
        self.prenet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 224 * 3, 1000),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000)
        )

        self.predict_net = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.prenet(x)
        return torch.sigmoid(self.predict_net(x))


class MResNet(nn.Module):
    def __init__(self, dtype='Res18', pretrain=True):
        super(MResNet, self).__init__()
        if dtype == 'Res34':
            self.prenet = models.resnet34(pretrained=pretrain)
        else:
            self.prenet = models.resnet18(pretrained=pretrain)

        self.predict_net = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.prenet(x)
        return torch.sigmoid(self.predict_net(x))


class SNet(nn.Module):
    def __init__(self, dtype='v1', pretrain=True):
        super(SNet, self).__init__()
        if dtype == 'v1':
            self.prenet = models.squeezenet1_0(pretrained=pretrain)
        else:
            self.prenet = models.squeezenet1_1(pretrained=pretrain)

        self.predict_net = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.prenet(x)
        return torch.sigmoid(self.predict_net(x))


class VNet(nn.Module):
    def __init__(self, dtype='v1', pretrain=True):
        super(VNet, self).__init__()
        if dtype == 'v1':
            self.prenet = models.vgg11(pretrained=pretrain)
        else:
            self.prenet = models.vgg13(pretrained=pretrain)

        self.predict_net = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.prenet(x)
        return torch.sigmoid(self.predict_net(x))

class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.LSTM_encoder = nn.LSTM(num_layers=1, hidden_size=12, input_size=3)
        self.prenet1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.prenet2 = models.resnet18(pretrained=True)

        self.predict_net = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = x.view([-1, 3, 224*224])
        x = x.permute(2, 0, 1)
        x, (cn, hn) = self.LSTM_encoder(x)
        x = x.permute(1, 2, 0)
        x = x.reshape([-1, 12, 224, 224])
        x = self.prenet1(x)
        x = x.reshape([-1, 3, 224, 224])
        x = self.prenet2(x)
        return torch.sigmoid(self.predict_net(x))

if __name__ == "__main__":
    x = torch.rand([2, 3, 224, 224])
    print(x.shape)
    net = LSTMNet()
    y = net(x)
    print(y.shape)
