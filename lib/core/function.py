import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np

from lib.core.loss import loss_function_ab, loss_function_af, loss_function_bd
from lib.core.utils_box import reg2loc, bd2loc
from lib.core.ab_match import anchor_box_adjust, anchor_bboxes_encode
from lib.core.utils_ab import result_process_ab, result_process_af


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def ab_prediction_train(cfg, out_ab, label, boxes, action_num):
    '''
    Loss for anchor-based module includes: category classification loss, overlap loss and regression loss
    '''
    match_xs_ls = list()
    match_ws_ls = list()
    match_labels_ls = list()
    match_scores_ls = list()
    anchors_class_ls = list()
    anchors_overlap_ls = list()
    anchors_x_ls = list()
    anchors_w_ls = list()

    for i, layer_name in enumerate(cfg.MODEL.LAYERS_NAME):
        match_xs, match_ws, match_scores, match_labels, \
        anchors_x, anchors_w, anchors_overlap, anchors_class = \
            anchor_bboxes_encode(cfg, out_ab[i], label, boxes, action_num, layer_name)

        match_xs_ls.append(match_xs)
        match_ws_ls.append(match_ws)
        match_scores_ls.append(match_scores)
        match_labels_ls.append(match_labels)

        anchors_x_ls.append(anchors_x)
        anchors_w_ls.append(anchors_w)
        anchors_overlap_ls.append(anchors_overlap)
        anchors_class_ls.append(anchors_class)

    # collect the predictions
    match_xs_ls = torch.cat(match_xs_ls, dim=1)
    match_ws_ls = torch.cat(match_ws_ls, dim=1)
    match_labels_ls = torch.cat(match_labels_ls, dim=1)
    match_scores_ls = torch.cat(match_scores_ls, dim=1)
    anchors_class_ls = torch.cat(anchors_class_ls, dim=1)
    anchors_overlap_ls = torch.cat(anchors_overlap_ls, dim=1)
    anchors_x_ls = torch.cat(anchors_x_ls, dim=1)
    anchors_w_ls = torch.cat(anchors_w_ls, dim=1)

    return anchors_x_ls, anchors_w_ls, anchors_overlap_ls, anchors_class_ls, \
           match_xs_ls, match_ws_ls, match_scores_ls, match_labels_ls


def ab_predict_eval(cfg, out_ab):
    # collect predictions
    anchors_class_ls = list()
    anchors_overlap_ls = list()
    anchors_x_ls = list()
    anchors_w_ls = list()

    for i, layer_name in enumerate(cfg.MODEL.LAYERS_NAME):
        anchors_class, anchors_overlap, anchors_x, anchors_w = anchor_box_adjust(cfg, out_ab[i], layer_name)
        anchors_class_ls.append(anchors_class)
        anchors_overlap_ls.append(anchors_overlap)
        anchors_x_ls.append(anchors_x)
        anchors_w_ls.append(anchors_w)

    # classification score
    anchors_class_ls = torch.cat(anchors_class_ls, dim=1)
    # overlap
    anchors_overlap_ls = torch.cat(anchors_overlap_ls, dim=1)
    # regression
    anchors_x_ls = torch.cat(anchors_x_ls, dim=1)
    anchors_w_ls = torch.cat(anchors_w_ls, dim=1)

    return anchors_class_ls, anchors_overlap_ls, anchors_x_ls, anchors_w_ls


def train(cfg, train_loader, model, optimizer, ema_model=None, epoch=0):
    model.train()
    loss_record = 0

    for iter, (feat_spa, feat_tem, boxes, label, action_num, cls_label, reg_label, cate_label, bd_base, bd_refine) in enumerate(train_loader):
        optimizer.zero_grad()

        feature = torch.cat((feat_spa, feat_tem), dim=1)
        feature = feature.type_as(dtype)
        boxes = boxes.float().type_as(dtype)
        label = label.type_as(dtypel)
        # af label
        # we do not calculate binary classification loss for anchor-free branch
        reg_label = reg_label.type_as(dtype)
        cate_label = cate_label.type_as(dtype)

        bd_base = bd_base.type_as(dtype)
        bd_refine = bd_refine.type_as(dtype)

        batch_size = cate_label.size(0)
        vid_label = torch.zeros((batch_size, 21))
        pos_mask = cate_label.nonzero()
        vid_label[pos_mask[:, 0], cate_label[pos_mask[:, 0], pos_mask[:, 1]].long()] = 1
        vid_label = vid_label.type_as(dtype)

        out_bd, vid_preds = model(feature, vid_labels=vid_label)

        preds_cls, preds_reg, preds_bd = out_bd
        cate_loss_af, bd_cls_loss, bd_reg_loss, vid_loss = loss_function_bd(cate_label, preds_cls, bd_base, preds_bd, bd_refine, preds_reg, vid_label, vid_preds, cfg)

        reg_weight = bd_cls_loss.detach() / max(bd_reg_loss.item(), 0.01)

        loss = cate_loss_af + bd_cls_loss + bd_reg_loss + vid_loss

        loss.backward()
        optimizer.step()
        loss_record = loss_record + loss.item()
        if ema_model is not None:
            ema_model.update(model)
        
        if (iter + 1) % 10 == 0:
            print('epoch: %d iter: %d cate_loss_af: %.6f bd_cls_loss: %.4f bd_reg_loss: %.6f vid_loss: %.4f' % (epoch, iter, cate_loss_af, bd_cls_loss, bd_reg_loss, vid_loss))
            with open(os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.LOG_FILE), 'a') as f:
                f.write('epoch: %d iter: %d cate_loss_af: %.6f bd_cls_loss: %.4f bd_reg_loss: %.6f vid_loss: %.4f \n' % (epoch, iter, cate_loss_af, bd_cls_loss, bd_reg_loss, vid_loss))

    loss_avg = loss_record / len(train_loader)
    return loss_avg


def evaluation(val_loader, model, epoch, cfg):
    model.eval()

    out_df_af = pd.DataFrame(columns=cfg.TEST.OUTDF_COLUMNS_AF)
    out_df_af_cc = pd.DataFrame(columns=cfg.TEST.OUTDF_COLUMNS_AF)
    for feat_spa, feat_tem, begin_frame, video_name in val_loader:
        begin_frame = begin_frame.detach().numpy()

        feature = torch.cat((feat_spa, feat_tem), dim=1)
        feature = feature.type_as(dtype)
        out_bd, vid_preds = model(feature)

        ################################ bdnet ###############################
        l_ss = ((64, 1), (32, 2), (16, 4), (8, 8), (4, 16), (2, 32))
        len_factor = []
        for ls in l_ss:
            tmp_factor = [[ls[1], ls[1]]] * ls[0]
            len_factor.extend(tmp_factor)
        len_factor = torch.from_numpy(np.array(len_factor))

        preds_cls, preds_reg, preds_bd = out_bd
        len_factor = len_factor.to(preds_reg.device)
        # preds_reg *= len_factor
        m = nn.Softmax(dim=2).cuda()
        preds_cls = m(preds_cls)
        # preds_cls = preds_cls.sigmoid()
        preds_loc, bd_score = bd2loc(cfg, preds_bd, preds_reg, len_factor)
        # preds_loc = reg2loc(cfg, preds_reg)

        preds_cls = preds_cls.detach().cpu().numpy()

        xmins = preds_loc[:, :, 0]
        xmins = xmins.detach().cpu().numpy()
        xmaxs = preds_loc[:, :, 1]
        xmaxs = xmaxs.detach().cpu().numpy()

        tmp_df_af, tmp_df_af_cc = result_process_af(video_name, begin_frame, preds_cls, xmins, xmaxs, cfg, bd_score, vid_preds)
        out_df_af = pd.concat([out_df_af, tmp_df_af], sort=True)
        out_df_af_cc = pd.concat([out_df_af_cc, tmp_df_af_cc], sort=True)
        ################################ bdnet ###############################

    if cfg.BASIC.SAVE_PREDICT_RESULT:

        predict_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.PREDICT_CSV_FILE + '_af' + str(epoch) + '.csv')
        print('predict_file', predict_file)
        out_df_af.to_csv(predict_file, index=False)

    return out_df_af, out_df_af_cc

