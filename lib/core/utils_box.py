import math
import numpy as np
import torch

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


import math
import numpy as np
import torch

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def locate_point(tem_length, stride):
    location = list()
    stride_l = math.floor(stride / 2)
    for i in range(tem_length):
        data = i * stride + stride_l
        location.append(data)
    return location


def location_point_back(cfg, to_tensor=False, to_duration=False):
    location = list()
    for layer_name in cfg.MODEL.LAYERS_NAME:
        tem_length = cfg.MODEL.TEMPORAL_LENGTH[layer_name]
        stride = cfg.MODEL.TEMPORAL_STRIDE[layer_name]
        loc_layer = locate_point(tem_length, stride)
        location.extend(loc_layer)

    if to_duration:
        duration = list()
        start_time = 0
        end_time = 2
        for layer_name in cfg.MODEL.LAYERS_NAME:
            tem_length = cfg.MODEL.TEMPORAL_LENGTH[layer_name]
            data = [[start_time, end_time] for i in range(tem_length)]
            duration.extend(data)
            # updata
            start_time = end_time
            end_time = end_time * 2
        print(len(duration))

    # convert to tensor
    if to_tensor:
        location_point = np.array(location)
        location_point = torch.from_numpy(location_point)
        location_point = torch.unsqueeze(location_point, dim=0)
        location_point = location_point.type_as(dtype)
        return location_point
    elif to_duration:
        return location, duration
    else:
        return location


def reg2loc(cfg, pred_regs):
    # pred_regs [batch, 254, 2]
    num_batch = pred_regs.size()[0]
    location_point = location_point_back(cfg, to_tensor=True)  # [1, 254]
    location_point = location_point.repeat(num_batch, 1)
    pred_locs = torch.zeros(pred_regs.size()).type_as(dtype)

    # filter out
    num_pred = pred_regs.size(1)
    location_point = location_point[:, :num_pred].contiguous()
    # left boundary
    pred_locs[:, :, 0] = location_point - pred_regs[:, :, 0]
    # right boundary
    pred_locs[:, :, 1] = location_point + pred_regs[:, :, 1]

    return pred_locs

def bd2loc(cfg, preds_bd, preds_reg, len_factor):
    len_factor = len_factor[:, 0]
    num_batch = preds_reg.size()[0]
    location_point = location_point_back(cfg, to_tensor=True)  # [1, 254]
    location_point = location_point.repeat(num_batch, 1)
    pred_locs = torch.zeros(preds_reg.size()).type_as(dtype)

    preds_bd_left = preds_bd[..., :cfg.MODEL.BIN]
    preds_bd_right = preds_bd[..., cfg.MODEL.BIN:]
    preds_reg_left = preds_reg[..., :cfg.MODEL.BIN]
    preds_reg_right = preds_reg[..., cfg.MODEL.BIN:]

    left_idx = preds_bd_left.argmax(dim=-1)
    right_idx = preds_bd_right.argmax(dim=-1)

    left_reg = preds_reg_left[:, range(left_idx.shape[1]), left_idx[0]]
    right_reg = preds_reg_right[:, range(right_idx.shape[1]), right_idx[0]]

    left_score = preds_bd_left.sigmoid().max(dim=-1)[0]
    right_score = preds_bd_right.sigmoid().max(dim=-1)[0]
    # bd_score = (left_score + right_score) / 2
    bd_score = left_score * right_score

    left_idx = left_idx.squeeze()
    right_idx = right_idx.squeeze()
    left_reg = left_reg.squeeze()
    right_reg = right_reg.squeeze()

    center_left = left_idx * len_factor * cfg.MODEL.BS + cfg.MODEL.BS * len_factor / 2
    center_right = right_idx * len_factor * cfg.MODEL.BS + cfg.MODEL.BS * len_factor / 2

    final_left = center_left + left_reg
    final_right = center_right + right_reg
    final_left = final_left.unsqueeze(0)
    final_right = final_right.unsqueeze(0)

    # filter out
    num_pred = preds_reg.size(1)
    location_point = location_point[:, :num_pred].contiguous()
    # left boundary
    pred_locs[:, :, 0] = location_point - final_left
    # right boundary
    pred_locs[:, :, 1] = location_point + final_right

    return pred_locs, bd_score


if __name__ == '__main__':
    from config import cfg, update_config
    cfg_file = '/data/2/v-yale/ActionLocalization/experiments/anet/ssad.yaml'
    update_config(cfg_file)

    location, duration = location_point_back(cfg, to_tensor=False, to_duration=True)
    i = 0
    for loc, dur in zip(location, duration):
        print(i, loc, dur)
        i += 1


