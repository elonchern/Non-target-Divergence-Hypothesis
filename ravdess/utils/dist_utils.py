from torch.autograd import Function
import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Adapted from https://github.com/declare-lab/MISA/blob/master/src/utils/functions.py
"""


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)





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
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

class DKD(nn.Module):
    def __init__(self):
        super(DKD, self).__init__()

    def forward(self, logits_student, logits_teacher, target, alpha, beta, temperature,omega):
        # from paper "Decoupled Knowledge Distillation"
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
    
        
        tckd_kl = F.kl_div(log_pred_student, pred_teacher, reduction='none')
        weight_tckd_kl = tckd_kl *  (omega[:,None])
        total_tckd_kl = torch.sum(weight_tckd_kl)
        
        
        tckd_loss = (total_tckd_kl* (temperature**2)/ target.shape[0])
        
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        
       
        
        
        nckd_kl = F.kl_div(log_pred_student_part2 , pred_teacher_part2, reduction='none',)
        
        weight_nckd_kl = nckd_kl * (omega[:,None])
        total_nckd_kl = torch.sum(weight_nckd_kl)  
        
        nckd_loss = (total_nckd_kl * (temperature**2) / target.shape[0])

        return alpha * tckd_loss + beta * nckd_loss
    
    
def pd_diff(pl,y_data,y_pred):
    gt_mask = _get_gt_mask(y_pred, y_data)
    other_mask = _get_other_mask(y_pred, y_data)
    pred_student = F.softmax(y_pred , dim=1)
    pred_teacher = F.softmax(pl , dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    
    
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
    
    return omega
    
    
