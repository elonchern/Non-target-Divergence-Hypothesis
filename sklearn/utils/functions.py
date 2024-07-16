### Some functions
import torch  
import numpy as np
import torch.nn as nn 
from torch.nn import functional as F
def predict(x, m):
    m.eval()
    # Convert into numpy element to tensor
    x = torch.from_numpy(x).type(torch.FloatTensor)
    # Predict and return ans
    output = m(x, 0)
    _, predicted = torch.max(output, 1)
    return predicted.numpy()



def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt 



def pd_diff(pl,y_data,y_pred):
    gt_mask = _get_gt_mask(y_pred, y_data)
    other_mask = _get_other_mask(y_pred, y_data)
    pred_student = F.softmax(y_pred , dim=1)
    pred_teacher = F.softmax(pl , dim=1)
    label = F.one_hot(y_data, 4)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    label_mask = cat_mask(label, gt_mask, other_mask)

    pred_teacher_part2 = F.softmax(
        pl  - 1000.0 * gt_mask, dim=1
    )
    pred_student_part2 = F.softmax(
        y_pred  - 1000.0 * gt_mask, dim=1
    )
    
    
    max_value, max_index_t = torch.max(torch.softmax(pl, dim=1), dim=1)
    mask_tf = max_index_t - y_data
    mask_tf[mask_tf!=0] = True # teacher 预测错误的地方
    mask_tt = 1 - mask_tf
    
    max_index_s = torch.argmax(y_pred, dim=1)
    mask_sf = max_index_s - y_data
    mask_sf[mask_sf!=0] = True # student 预测错误的地方
    mask_st = 1 - mask_sf
     
    # 计算余弦距离
    cosine_similarity = torch.cosine_similarity(pl,y_pred, dim=1)

    normalized_similarity = (cosine_similarity + 1) / 2
    
    cosine_mask_g = cosine_similarity.ge(0.996) # 大于0.996保留, 小于0.996置为0
    
    mask = normalized_similarity*cosine_mask_g
    
    omega = mask_tf*mask + mask_tt
    
    return omega, mask_tf, pred_student, pred_teacher #label_mask.float()

class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        
        s_N, s_C = s_shape
        t_N, t_C = t_shape
        assert s_C == t_C

        self.bn = nn.BatchNorm1d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)

    


    
    