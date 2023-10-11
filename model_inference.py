import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from lib.config import cfg
from lib.utils.utils import fix_random_seed
from lib.config import update_config
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from lib.dataset.TALDataset import TALDataset
from lib.models.bdrcnet import LocNet
from lib.core.function import evaluation
from lib.core.post_process import final_result_process
from lib.core.utils_ab import weight_init


def parse_args():
    parser = argparse.ArgumentParser(description='SSAD temporal action localization')
    parser.add_argument('--cfg', type=str, help='experiment config file', default='./experiments/A2Net_thumos.yaml')
    parser.add_argument('--resume', type=str, help='model path', default='./output/thumos_cc/model_34.pth')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(args.cfg)
    # create output directory
    if cfg.BASIC.CREATE_OUTPUT_DIR:
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    fix_random_seed(cfg.BASIC.SEED)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE

    # data loader
    val_dset = TALDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False, drop_last=False, num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)

    model = LocNet(cfg)
    model.apply(weight_init)
    model.cuda()

    if args.resume is None:
        return
    
    ckpt = torch.load(args.resume)['model']
    model.load_state_dict(ckpt)
    out_df_af, out_df_af_cc = evaluation(val_loader, model, 0, cfg)
    out_df_list = out_df_af
    final_result_process(out_df_list, 0, cfg, flag=0)
    out_df_list = out_df_af_cc
    final_result_process(out_df_list, 0, cfg, flag=1)


if __name__ == '__main__':
    main()
