import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.blstm = nn.LSTM(513*2, 400, dropout=0.1, num_layers=2, bidirectional=True, batch_first=True)
        self.LReLU = nn.LeakyReLU(0.3)
        self.Dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(400*2,600)
        self.fc2 = nn.Linear(600,513)
    def forward(self,x,y):
        #  x: clean mag, y: noise mag
        inputs = torch.cat((x,y),dim=2)
        output, _ = self.blstm(inputs)
        output = self.fc1(output)
        output = self.LReLU(output)
        output = self.Dropout(output)
        output = self.fc2(output)

        return torch.exp(1.5+4*torch.tanh(output))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(spectral_norm(nn.Conv2d(3, 8,(5,5))))
        layers.append(spectral_norm(nn.Conv2d(8, 16, (7,7))))
        layers.append(spectral_norm(nn.Conv2d(16, 32, (10,10))))
        layers.append(spectral_norm(nn.Conv2d(32, 48, (15,15))))
        layers.append(spectral_norm(nn.Conv2d(48, 64, (20,20))))
        self.layers = nn.ModuleList(layers)

        self.GAPool = nn.AdaptiveAvgPool2d((1,1))
        self.LReLU = nn.LeakyReLU(0.3)
        self.fc1 = spectral_norm(nn.Linear(64,64))
        self.fc2 = spectral_norm(nn.Linear(64,10))
        self.fc3 = spectral_norm(nn.Linear(10,2))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            x = self.LReLU(x)

        x = self.GAPool(x)
        B = x.shape[0]
        C = x.shape[1]

        x = x.view(B,C).contiguous()
        x = self.LReLU(self.fc1(x))
        x = self.LReLU(self.fc2(x))

        x = torch.sigmoid(self.fc3(x))
        return x


        

        
