import torch
import torch.nn.functional as F

class DKDLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, temperature=1.0):
        super(DKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def forward(self, logits_student, logits_teacher, target):
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        pred_student = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)

        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='sum')
            * (self.temperature**2)
            / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1
        )

        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
            * (self.temperature**2)
            / target.shape[0]
        )

        return self.alpha * tckd_loss + self.beta * nckd_loss

if __name__ == '__main__':
    # 使用示例
    # 创建损失函数实例
    loss_func = DKDLoss(alpha=1.0, beta=1.0, temperature=1.0)

    # 假设有学生和教师的logits以及目标标签
    logits_student = torch.randn(10, 5)  # 假设有10个样本，5个类别
    logits_teacher = torch.randn(10, 5)
    target = torch.randint(0, 5, (10,))

    # 计算损失
    loss = loss_func(logits_student, logits_teacher, target)
    print(loss)