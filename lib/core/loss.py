import torch
import torch.nn as nn
import torch.nn.functional as F


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def abs_smooth(x):
    '''
    Sommth L1 loss
    Defined as:
        x^2 / 2        if abs(x) < 1
        abs(x) - 0.5   if abs(x) >= 1
    '''
    absx = torch.abs(x)
    minx = torch.min(absx, torch.tensor(1.0).type_as(dtype))
    loss = 0.5 * ((absx - 1) * minx + absx)
    loss = loss.sum()
    return loss


def loss_function_ab(anchors_x, anchors_w, anchors_overlap, anchors_class, match_x, match_w, match_scores, match_labels, cfg):
    '''
    calculate classification loss, localization loss and overlap_loss
    pmask, hmask and nmask are used to select training samples
    '''

    anchors_xmin = anchors_x - anchors_w / 2
    anchors_xmax = anchors_x + anchors_w / 2
    match_xmin = match_x - match_w / 2
    match_xmax = match_x + match_w / 2

    anchors_xmin = anchors_xmin.view(-1)
    anchors_xmax = anchors_xmax.view(-1)
    match_xmin = match_xmin.view(-1)
    match_xmax = match_xmax.view(-1)
    match_scores = match_scores.view(-1)
    anchors_overlap = anchors_overlap.view(-1)
    anchors_class = anchors_class.view(-1, cfg.DATASET.NUM_CLASSES)
    match_labels = match_labels.view(-1)

    pmask = torch.gt(match_scores, cfg.TRAIN.MATCH_TH).type_as(dtype)
    num_positive = pmask.sum().item()
    num_entries = match_scores.size()[-1]

    hmask = torch.le(match_scores, cfg.TRAIN.MATCH_TH)
    hmask = hmask & torch.gt(anchors_overlap, cfg.TRAIN.MATCH_TH)
    hmask = hmask.type_as(dtype)
    num_hard = hmask.sum().item()

    # negative ratio: the ratio used to choose easy negative anchors
    if (num_positive > 0) and (num_entries > num_positive+num_hard):
        # r_negative = (cfg.TRAIN.NEGATIVE_RATIO - num_hard/num_positive) * num_positive\
        #              / (num_entries - num_positive - num_hard)
        r_negative = cfg.TRAIN.NEGATIVE_RATIO * num_positive / (num_entries - num_positive - num_hard)
    else:
        # if no positive example, select 50% negative examples
        r_negative = 0.5
    r_negative = min(r_negative, 1)

    nmask = torch.FloatTensor(pmask.size()).uniform_().type_as(dtype)
    nmask = nmask * (1.0 - pmask)
    nmask = nmask * (1.0 - hmask)
    nmask = torch.gt(nmask, 1.-r_negative).type_as(dtype)

    # classification loss
    weights = pmask + nmask + hmask
    # print('cls weights', weights)
    keep = weights == 1.0
    anchors_class = anchors_class[keep, :]
    match_labels = match_labels[keep]
    cls_loss_f = torch.nn.CrossEntropyLoss()
    cls_loss = cls_loss_f(anchors_class, match_labels)

    # overlap loss
    weights = pmask + nmask + hmask
    # print('weights', weights.sum().item())
    keep = weights == 1.0
    match_scores = match_scores[keep]
    anchors_overlap = anchors_overlap[keep]
    overlap_loss = abs_smooth(match_scores - anchors_overlap)

    # localization loss
    if torch.sum(pmask) > 0:
        weights = pmask
        keep = weights == 1.0
        anchors_xmin = anchors_xmin[keep]
        match_xmin = match_xmin[keep]
        anchors_xmax = anchors_xmax[keep]
        match_xmax = match_xmax[keep]
        loc_loss = abs_smooth(anchors_xmin - match_xmin) + abs_smooth(anchors_xmax - match_xmax)
    else:
        loc_loss = torch.tensor(0.).type_as(overlap_loss)
    # print('loss:', cls_loss.item(), loc_loss.item(), overlap_loss.item())

    return cls_loss, overlap_loss, loc_loss


def sel_fore_reg(cls_label_view, target_regs, pred_regs):
    sel_mask = cls_label_view >= 1.0
    target_regs_view = target_regs.view(-1)
    target_regs_sel = target_regs_view[sel_mask]
    pred_regs_view = pred_regs.view(-1)
    pred_regs_sel = pred_regs_view[sel_mask]

    return target_regs_sel, pred_regs_sel


def loss_function_af(cate_label, preds_cls, target_regs_batch, pred_regs_batch, cfg):
    '''
    calculate fore-/back- classification loss, regression loss
    '''
    # we should filter out redundant label
    num_pred = preds_cls.size(1)
    cate_label = cate_label[:, :num_pred].contiguous()
    target_regs_batch = target_regs_batch[:, :num_pred, :].contiguous()

    cate_label_view = cate_label.view(-1)
    cate_label_view = cate_label_view.type_as(dtypel)

    preds_cls_view = preds_cls.view(-1, 21)
    # cls loss
    cate_loss_f = torch.nn.CrossEntropyLoss()
    cate_loss = cate_loss_f(preds_cls_view, cate_label_view)
    # cate_new_label = F.one_hot(cate_label_view, 21).type_as(dtypel)
    # cate_new_label = cate_new_label[..., 1:]
    # cate_loss = sigmoid_focal_loss(preds_cls_view, cate_new_label, reduction='none')
    # cate_loss = cate_loss.sum() / 100

    # regression loss
    target_regs_left_sel, pred_regs_left_sel = sel_fore_reg(cate_label_view, target_regs_batch[:, :, 0], pred_regs_batch[:, :, 0])
    target_regs_right_sel, pred_regs_right_sel = sel_fore_reg(cate_label_view, target_regs_batch[:, :, 1], pred_regs_batch[:, :, 1])

    # regression with smooth L1 loss
    reg_loss_left = abs_smooth(target_regs_left_sel - pred_regs_left_sel)
    reg_loss_right = abs_smooth(target_regs_right_sel - pred_regs_right_sel)
    reg_loss = reg_loss_left + reg_loss_right

    return cate_loss, reg_loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def sel_fore_bd(bd_base, bd_refine, preds_reg):
    target_regs = bd_refine[bd_base >= 0.5]
    pred_regs = preds_reg[bd_base >= 0.5]
    return target_regs, pred_regs


def loss_function_bd(cate_label, preds_cls, bd_base, preds_bd, bd_refine, preds_reg, vid_label, vid_preds, cfg):
    '''
    calculate fore-/back- classification loss, regression loss
    '''
    loss_normalizer = 90
    # we should filter out redundant label
    num_pred = preds_cls.size(1)
    cate_label = cate_label[:, :num_pred].contiguous()
    bd_base = bd_base[:, :num_pred, :].contiguous()
    bd_refine = bd_refine[:, :num_pred, :].contiguous()

    cate_label_view = cate_label.view(-1)
    cate_label_view = cate_label_view.type_as(dtypel)

    preds_cls_view = preds_cls.view(-1, 21)
    # cls loss
    weight = torch.ones(21)
    weight[1:] = 5
    weight = weight.to(preds_cls_view.device)
    cate_loss_f = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
    cate_loss = cate_loss_f(preds_cls_view, cate_label_view)
    cate_loss = cate_loss.sum()
    cate_loss = cate_loss / loss_normalizer

    bd_base = bd_base.view(-1)
    preds_bd = preds_bd.view(-1)
    bd_refine = bd_refine.view(-1)
    preds_reg = preds_reg.view(-1)
    # bd cls loss
    bd_cls_gt = bd_base.view(-1, cfg.MODEL.BIN * 2)[cate_label_view > 0]
    bd_cls_pred = preds_bd.view(-1, cfg.MODEL.BIN * 2)[cate_label_view > 0]
    bd_cls_loss = sigmoid_focal_loss(bd_cls_pred, bd_cls_gt, reduction='none')
    bd_cls_loss = bd_cls_loss.sum() / loss_normalizer

    # bd regg loss
    target_regs, pred_regs = sel_fore_bd(bd_base, bd_refine, preds_reg)
    bd_reg_loss = abs_smooth(target_regs - pred_regs)
    bd_reg_loss = bd_reg_loss / loss_normalizer

    '''
    cc_module loss
    '''
    ce_criterion = nn.BCELoss(reduction='none')
    loss_vid = ce_criterion(vid_preds, vid_label[:, 1:].detach())
    loss_vid = loss_vid.sum() / loss_normalizer

    return cate_loss, bd_cls_loss, bd_reg_loss, loss_vid
