import torch.nn as nn
import torch


def conv1d_c512(cfg, stride, out_channels=None):
    if out_channels is None:
        out_channels = 512
    return nn.Conv1d(in_channels=512, out_channels=out_channels,
                     kernel_size=3, stride=stride, padding=1, bias=True)


class BaseFeatureNet(nn.Module):
    '''
    calculate feature
    input: [batch_size, 128, 1024]
    output: [batch_size, 32, 512]
    '''
    def __init__(self, cfg):
        super(BaseFeatureNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.IN_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.max_pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fea = self.relu(self.conv1(x))
        fea = self.relu(self.conv2(fea))
        fea = self.max_pooling(fea)
        return fea


class FeatNet(nn.Module):
    '''
    main network
    input: base feature, [batch_size, 32, 512]
    output: MAL1, MAL2, MAL3
    '''
    def __init__(self, cfg):
        super(FeatNet, self).__init__()
        self.base_feature_net = BaseFeatureNet(cfg)
        self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM,
                               out_channels=cfg.MODEL.CON1_FEAT_DIM,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.CON1_FEAT_DIM,
                               out_channels=cfg.MODEL.CON2_FEAT_DIM,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv1d(in_channels=cfg.MODEL.CON2_FEAT_DIM,
                               out_channels=cfg.MODEL.CON3_FEAT_DIM,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4 = nn.Conv1d(in_channels=cfg.MODEL.CON3_FEAT_DIM,
                               out_channels=cfg.MODEL.CON4_FEAT_DIM,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv5 = nn.Conv1d(in_channels=cfg.MODEL.CON4_FEAT_DIM,
                               out_channels=cfg.MODEL.CON5_FEAT_DIM,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv6 = nn.Conv1d(in_channels=cfg.MODEL.CON5_FEAT_DIM,
                               out_channels=cfg.MODEL.CON6_FEAT_DIM,
                               kernel_size=3, stride=2, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        base_feature = self.base_feature_net(x)
        mal1 = self.relu(self.conv1(base_feature))
        mal2 = self.relu(self.conv2(mal1))
        mal3 = self.relu(self.conv3(mal2))
        mal4 = self.relu(self.conv4(mal3))
        mal5 = self.relu(self.conv5(mal4))
        mal6 = self.relu(self.conv6(mal5))

        return mal1, mal2, mal3, mal4, mal5, mal6


############### anchor-free ##############
class PredHeadBranch(nn.Module):
    '''
    prediction module branch
    Output number is 2
      Regression: distance to left boundary, distance to right boundary
      Classification: probability
    '''
    def __init__(self, cfg, pred_channels=2):
        super(PredHeadBranch, self).__init__()
        self.conv1 = conv1d_c512(cfg, stride=1)
        self.conv2 = conv1d_c512(cfg, stride=1)
        self.pred = conv1d_c512(cfg, stride=1, out_channels=pred_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat1 = self.relu(self.conv1(x))
        feat2 = self.relu(self.conv2(feat1))
        pred = self.pred(feat2)  # [batch, 2, temporal_length]
        pred = pred.permute(0, 2, 1).contiguous()

        return pred

class BDHeadBranch(nn.Module):
    '''
    prediction module branch
    Output number is 2
      Regression: distance to left boundary, distance to right boundary
      Classification: probability
    '''
    def __init__(self, cfg, pred_channels=8):
        super(BDHeadBranch, self).__init__()
        self.conv1 = conv1d_c512(cfg, stride=1)
        self.conv2 = conv1d_c512(cfg, stride=1)
        self.pred_cls = conv1d_c512(cfg, stride=1, out_channels=pred_channels)
        self.pred = conv1d_c512(cfg, stride=1, out_channels=pred_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat1 = self.relu(self.conv1(x))
        feat2 = self.relu(self.conv2(feat1))
        pred = self.pred(feat2)  # [batch, 8, temporal_length]
        pred_cls = self.pred_cls(feat2)
        pred = pred.permute(0, 2, 1).contiguous()
        pred_cls = pred_cls.permute(0, 2, 1).contiguous()

        return pred_cls, pred


class PredHead(nn.Module):
    '''
    Predict classification and regression
    '''
    def __init__(self, cfg):
        super(PredHead, self).__init__()
        self.cls_branch = PredHeadBranch(cfg, pred_channels=cfg.DATASET.NUM_CLASSES)
        self.reg_branch = BDHeadBranch(cfg, pred_channels=cfg.MODEL.BIN * 2)
        # self.reg_branch = PredHeadBranch(cfg)

    def forward(self, x):
        # predict the probability for foreground or background
        cls = self.cls_branch(x)
        bd_cls, reg = self.reg_branch(x)
        # reg = self.reg_branch(x)

        return cls, reg, bd_cls
        # return cls, reg


class Scale(nn.Module):
    '''
    Different layers regression to different size range
    Learn a trainable scalar to automatically adjust the base of exp(si * x)
    '''
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ReduceChannel(nn.Module):
    def __init__(self, cfg):
        super(ReduceChannel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.CON1_FEAT_DIM, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.CON2_FEAT_DIM, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.conv3 = nn.Conv1d(in_channels=cfg.MODEL.CON3_FEAT_DIM, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.conv4 = nn.Conv1d(in_channels=cfg.MODEL.CON4_FEAT_DIM, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.conv5 = nn.Conv1d(in_channels=cfg.MODEL.CON5_FEAT_DIM, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.conv6 = nn.Conv1d(in_channels=cfg.MODEL.CON6_FEAT_DIM, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_list):
        mal1, mal2, mal3, mal4, mal5, mal6 = feat_list
        mal1 = self.relu(self.conv1(mal1))
        mal2 = self.relu(self.conv2(mal2))
        mal3 = self.relu(self.conv3(mal3))
        mal4 = self.relu(self.conv4(mal4))
        mal5 = self.relu(self.conv5(mal5))
        mal6 = self.relu(self.conv6(mal6))

        return mal1, mal2, mal3, mal4, mal5, mal6


class LocNetBD(nn.Module):
    '''
    Predict action boundary, based on features from different FPN levels
    '''
    def __init__(self, cfg):
        super(LocNetBD, self).__init__()
        # self.features = FeatNet(cfg)
        self.reduce_channels = ReduceChannel(cfg)
        self.pred = PredHead(cfg)

        self.scale0 = Scale()
        self.scale1 = Scale()
        self.scale2 = Scale()
        self.scale3 = Scale()
        self.scale4 = Scale()
        self.scale5 = Scale()
        self.scale6 = Scale()

    def _layer_cal(self, feat_list, scale_list):
        pred_cls = list()
        pred_reg = list()
        pred_bd = list()

        for feat, scale in zip(feat_list, scale_list):
            cls_tmp, reg_tmp, bd_cls_tmp = self.pred(feat)
            # cls_tmp, reg_tmp = self.pred(feat)
            reg_tmp = scale(reg_tmp)
            pred_cls.append(cls_tmp)
            pred_reg.append(reg_tmp)
            pred_bd.append(bd_cls_tmp)

        predictions_cls = torch.cat(pred_cls, dim=1)
        predictions_reg = torch.cat(pred_reg, dim=1)
        predictions_bd = torch.cat(pred_bd, dim=1)

        return predictions_cls, predictions_reg, predictions_bd
        # return predictions_cls, predictions_reg

    def forward(self, features_list):
        features_list = self.reduce_channels(features_list)
        scale_list = [self.scale0, self.scale1, self.scale2, self.scale3, self.scale4, self.scale5, self.scale6]

        predictions_cls, predictions_reg, predictions_bd = self._layer_cal(features_list, scale_list)
        # predictions_cls, predictions_reg = self._layer_cal(features_list, scale_list)

        return predictions_cls, predictions_reg, predictions_bd
        # return predictions_cls, predictions_reg


class CC_Module(nn.Module):
    def __init__(self, len_feature=2048, num_classes=20):
        super(CC_Module, self).__init__()
        self.len_feature = len_feature
        mid_dim = 512
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=mid_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=mid_dim, out_channels=num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(p=0.7)
        self.r_act = 8

    def forward(self, x, vid_labels=None):
        if vid_labels is not None:
            vid_labels = vid_labels[:, 1:]
        num_segments = x.shape[2]
        k_act = max(num_segments // self.r_act, 1)
        # x: (B, F, T)
        out = x
        # out: (B, F, T)
        out = self.conv_1(out)
        if vid_labels is not None:
            out = self.drop_out(out)
        cas = self.classifier(out)
        cas = cas.permute(0, 2, 1)
        # out: (B, T, C + 1)

        cas_sigmoid = self.sigmoid(cas)
        value, _ = cas_sigmoid.sort(descending=True, dim=1)
        topk_scores = value[:,:k_act,:]
        if vid_labels is None:
            vid_score = torch.mean(topk_scores, dim=1)
        else:
            vid_score = (torch.mean(topk_scores, dim=1) * vid_labels) + (torch.mean(cas_sigmoid[:,:,:], dim=1) * (1 - vid_labels))

        return vid_score


class LocNet(nn.Module):
    def __init__(self, cfg):
        super(LocNet, self).__init__()
        self.features = FeatNet(cfg)
        self.bd = LocNetBD(cfg)
        self.cc_module = CC_Module()

    def forward(self, x, vid_labels=None):
        vid_preds = self.cc_module(x, vid_labels)
        features = self.features(x)
        out_bd = self.bd(features)
        return out_bd, vid_preds
