import torch
import torch.nn as nn
import torch.nn.functional as F

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

def js_divergence(logits_student, logits_teacher, target,temperature=1.0):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student, dim=1)
    pred_teacher = F.softmax(logits_teacher, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    log_pred_teacher = torch.log(pred_teacher)
    
    m = 0.5 * (pred_student + pred_teacher)
    
    tckl_sm = (
        F.kl_div(log_pred_student, m, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    tckl_tm = (
        F.kl_div(log_pred_teacher, m, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    
    tcjsd = 0.5 * (tckl_sm + tckl_tm)
    
    
    pred_teacher_part2 = F.softmax(
        logits_teacher - 1000.0 * gt_mask, dim=1
    )
    pred_student_part2 = F.softmax(
        logits_student - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student - 1000.0 * gt_mask, dim=1
    )
    log_pred_teacher_part2 = F.log_softmax(
        logits_teacher - 1000.0 * gt_mask, dim=1
    )
    
    m_part2 = 0.5 * (pred_teacher_part2 + pred_student_part2)
    nckl_sm = (
        F.kl_div(log_pred_student_part2, m_part2, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    
    nckl_tm = (
        F.kl_div(log_pred_teacher_part2, m_part2, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    
    ncjsd = 0.5 * (nckl_sm + nckl_tm)
    
    return tcjsd, ncjsd


if __name__ == "__main__":
    # Generate random logits for student and teacher models
    N, num_classes = 8, 5  # Example batch size and number of classes
    logits_student = torch.randn(N, num_classes, requires_grad=True)
    logits_teacher = torch.randn(N, num_classes, requires_grad=True)
    
    # Generate random targets
    target = torch.randint(0, num_classes, (N,))
    
    # Calculate JS divergence
    tcjsd, ncjsd = js_divergence(logits_student, logits_teacher, target)
    
    # Print the results
    print("Total Classwise JS Divergence (tcjsd):", tcjsd.item())
    print("Non-classwise JS Divergence (ncjsd):", ncjsd.item())



