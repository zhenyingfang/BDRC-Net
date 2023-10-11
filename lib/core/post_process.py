import pandas as pd
import os
from evalute_thumos import thumos_eval

from lib.core.nms import temporal_nms


def get_video_fps(video_name, cfg):
    # determine FPS
    if video_name in cfg.TEST.VIDEOS_25FPS:
        fps = 25
    elif video_name in cfg.TEST.VIDEOS_24FPS:
        fps = 24
    else:
        fps = 30
    return fps


def final_result_process(out_df, epoch, cfg, flag):
    if flag == 0:
        res_txt_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.PREDICT_TXT_FILE + '_' + str(epoch).zfill(2)+'.txt')
    else:
        res_txt_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.PREDICT_TXT_FILE + '_' + str(epoch).zfill(2)+'_cc.txt')
    if os.path.exists(res_txt_file):
        os.remove(res_txt_file)
    f = open(res_txt_file, 'a')

    df_af = out_df
    df_name = df_af

    video_name_list = list(set(df_name.video_name.values[:]))

    for video_name in video_name_list:
        tmpdf = df_af[df_af.video_name == video_name]

        # assign cliffDiving instance as diving too
        type_set = list(set(tmpdf.cate_idx.values[:]))
        if cfg.TEST.CATE_IDX_OCC in type_set:
            cliff_diving_df = tmpdf[tmpdf.cate_idx == cfg.TEST.CATE_IDX_OCC]
            diving_df = cliff_diving_df
            diving_df.loc[:, 'cate_idx'] = cfg.TEST.CATE_IDX_REP
            tmpdf = pd.concat(([tmpdf, diving_df]))

        df_nms = temporal_nms(tmpdf, cfg)

        # ensure there are most 200 proposals
        df_vid = df_nms.sort_values(by='score', ascending=False)
        fps = get_video_fps(video_name, cfg)

        for i in range(min(len(df_vid), cfg.TEST.TOP_K_RPOPOSAL)):
            start_time = df_vid.start.values[i] / fps
            end_time = df_vid.end.values[i] / fps
            label = df_vid.label.values[i]

            strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (video_name, float(start_time), float(end_time), label, df_vid.score.values[i])
            f.write(strout)

    f.close()
    dmap = thumos_eval(res_txt_file)
    with open(os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.LOG_FILE), 'a') as f:
        f.write("Rest: 0.1: %.4f 0.2: %.4f 0.3: %.4f 0.4: %.4f 0.5: %.4f 0.6: %.4f 0.7: %.4f 0.8: %.4f 0.9: %.4f avg: %.4f \n" % (dmap[0], dmap[1], dmap[2], dmap[3], dmap[4], dmap[5], dmap[6], dmap[7], dmap[8], dmap.mean()))
