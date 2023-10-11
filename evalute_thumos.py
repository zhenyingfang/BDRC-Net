import os
import json
import numpy as np
import pandas as pd
from metrics.eval_detection import ANETdetection


def result2json(result, class_dict):
    result_file = []    
    for i in range(len(result)):
        line = {'label': str(class_dict[int(result[i][0])]), 'score': result[i][1],
                'segment': [result[i][2], result[i][3]]}
        result_file.append(line)
    return result_file


def thumos_eval(pred_path):
    df_class = pd.read_csv('./data/thumos/detclasslist.txt', header=None, sep=' ')
    df_class.columns = ['id', 'name']
    dfc_ids = df_class['id'].values
    dfc_names = df_class['name'].values
    classlist = dict()
    name2id = dict()
    for dfci, dfcn in zip(dfc_ids, dfc_names):
        classlist[dfci] = dfcn
        name2id[dfcn] = dfci

    groundtruth_filename = './data/thumos/th14_groundtruth.json'

    save_dict = {'version': 'thumos14', 'external_data': 'None', 'results': dict()}

    df = pd.read_csv(pred_path, header=None, sep='	')
    df.columns = ['vid', 's', 'e', 'id', 'score']
    p_vid = df['vid'].values
    p_s = df['s'].values
    p_e = df['e'].values
    p_id = df['id'].values
    p_score = df['score'].values

    pred_dict = dict()
    for pv, ps, pe, pid, pscore in zip(p_vid, p_s, p_e, p_id, p_score):
        if pv in pred_dict.keys():
            pred_dict[pv].append([pid, pscore, ps, pe])
        else:
            pred_dict[pv] = [[pid, pscore, ps, pe]]

    vids = list(pred_dict.keys())
    for vname in vids:
        save_dict['results'][vname] = result2json(pred_dict[vname], classlist)

    tiou_thresholds = np.linspace(0.1, 0.9, 9)

    anet_detection = ANETdetection(groundtruth_filename,
                                    save_dict,
                                    subset='test',
                                    tiou_thresholds=tiou_thresholds,
                                    verbose=True,
                                    check_status=False)
    dmap = anet_detection.evaluate()
    return dmap
