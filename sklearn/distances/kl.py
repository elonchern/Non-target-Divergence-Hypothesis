import torch

from torch.nn import functional as F


def kl_distance(pl, y_pred,upper_bound, lower_bound):
    distances = F.kl_div(torch.log_softmax(pl, dim=1),torch.softmax(y_pred, dim=1), reduction='none',)
    dis_mean = torch.mean(distances, dim=1)
    # output = torch.sigmoid(distances)
    
    dis_mean[dis_mean<lower_bound] = 0
    
    out = torch.clamp(dis_mean/(upper_bound), max=1.0)
    return out.detach()