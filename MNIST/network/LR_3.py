import  torch
import  torch.nn as nn
from copy import deepcopy


def my_permute(x, index):  # generate a new tensor
    y = x.reshape(x.shape[0], -1).detach().clone()  # flatten all feature, this function will only be used in the
    # context of with no grad
    perm_index = torch.randperm(x.shape[0])
    for i in index:
        y[:, i] = y[perm_index, i]
    y = y.reshape(*x.size())  # reshape to original size
    return y


def my_permute_new(x, index):
    y = deepcopy(x)
    perm_index = torch.randperm(x.shape[0])
    for i in index:
        y[:, i] = x[perm_index, i]
    return y


def my_freeze(x, index):  # in place modification
    ori_size = x.size()
    x = x.reshape(x.shape[0], -1)
    x[:, index] = 0
    x = x.reshape(*ori_size)
    return x


def my_freeze_new(x, index):  # in place modification
    # y = deepcopy(x)
    # y = x
    y = x.clone()

    y[:, index] = 0
    # tmp_mean = x[:, index].mean(dim=0)
    # y[:, index] = tmp_mean

    return y


def my_change(x, change_type, index):
    if change_type == 'permute':
        return my_permute_new(x, index)
    elif change_type == 'freeze':
        return my_freeze_new(x, index)
    else:
        raise ValueError("Undefined change_type")
    
    
class MLP(nn.Module):

    def __init__(self,input_dim):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
          
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(inplace=True),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),    
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(256, 10),
            nn.LeakyReLU(inplace=True),
        )
        
    def get_feature_dim(self, place=None):
            if self.input_dim == 28*28:
                feature_dim_list = [28*28, 1024, 256, 10]
            elif self.input_dim == 28*28*3:
                
                feature_dim_list = [28*28*3, 1024, 256, 10]
            else:
                raise ValueError(f"Unsupported input_dim: {self.input_dim}")
            
            return feature_dim_list[place] if place else feature_dim_list       
        

    def forward(self, x,change_type=None, place=None, index=None):
        # change_type = 'permute' / 'freeze'
        if place == 0:
            x = my_change(x, change_type, index)
        
        x = self.fc1(x)    
        
        if place == 1:
            x = my_change(x, change_type, index)
        
        x = self.fc2(x)

        if place == 2:
            x = my_change(x, change_type, index)
            
        x = self.fc3(x)
        
        if place == 3:
            x = my_change(x, change_type, index)

        return x