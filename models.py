import torch
from torch import nn

class Single_layer(nn.Module):
    def __init__(self):
        super(Single_layer, self).__init__()
        self.module = nn.Sequential(
                    nn.Linear(26847, 82),
        )

    def forward(self,inp):
        x = self.module(inp)
        return x

class Standard(nn.Module):
    def __init__(self):
        
        super(Standard, self).__init__()
        self.module = nn.Sequential(
                    nn.Linear(26847, 1000),
                    torch.nn.BatchNorm1d(1000),
                    nn.Sigmoid(), 
                    #nn.Linear(1000, 100),
                    nn.Linear(1000,82),
        )

    def forward(self,inp):
        x = self.module(inp)
        return x


class Mini_hourglass(nn.Module):
    def __init__(self):
        super(Mini_hourglass, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(26847, 10),
            torch.nn.BatchNorm1d(10),
            nn.Sigmoid(), 
            nn.Linear(10, 82),
            torch.nn.BatchNorm1d(82),
            nn.LeakyReLU(), 
            nn.Dropout(p=0.5),
            nn.Linear(82, 82),
        )

    def forward(self,inp):
        x = self.module(inp)
        return x

from torch.utils.data import Dataset
import numpy as np

class Valid_Dataset(Dataset):
    def __init__(self, data, targets):

        self.data = data
        self.targets = targets

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, idx):
        inp = np.array(self.data[idx])

        target = np.array(self.targets[idx])

        inp = torch.from_numpy(inp)[0].float().cuda()  #Fix wierd dimention buig here
        target = torch.from_numpy(target).long().cuda()
        return inp, target