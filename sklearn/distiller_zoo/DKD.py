import torch
from torch.nn import functional as F
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


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature,omega):
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
    
    # numb_t = torch.sum(1-mask_t)
    tckd_loss = (total_tckd_kl* (temperature**2)/ target.shape[0])
    # print("tckd_loss={}".format(tckd_loss))
    
    # tckd_loss = (
    #     F.kl_div(log_pred_student, pred_teacher, reduction='sum')
    #     * (temperature**2)
    #     / target.shape[0]
    # )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    
    # max_value, max_index_t = torch.max(torch.softmax(input_mask, dim=1), dim=1)
    
    
    # nckd_kl = F.kl_div(log_pred_student_part2 * max_index_t[:,None], pred_teacher_part2* max_index_t[:,None], reduction='none',) #size_average=True
    nckd_kl = F.kl_div(log_pred_student_part2 , pred_teacher_part2, reduction='none',)
    
    weight_nckd_kl = nckd_kl * (omega[:,None])
    total_nckd_kl = torch.sum(weight_nckd_kl)  
    
    nckd_loss = (total_nckd_kl * (temperature**2) / target.shape[0])
    
    # nckd_loss = (
    #     F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
    #     * (temperature**2)
    #     / target.shape[0]
    # )
    
    
    return alpha * tckd_loss + beta * nckd_loss 