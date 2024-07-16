from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
import torch

def emd_distances(xa, xb):
    distances = []
    for i in range(len(xa)):
        distance = wasserstein_distance(xa[i].detach().numpy(), xb[i].detach().numpy())
        distances.append(distance)
    distance_torch = torch.tensor(distances)
    output = torch.sigmoid(distance_torch)
    output[output<0.02] = 0
    return output