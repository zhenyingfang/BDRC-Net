import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from lib.config import cfg
from lib.utils.utils import fix_random_seed
from lib.config import update_config
import pprint
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from lib.dataset.TALDataset import TALDataset
from lib.models.bdrcnet import LocNet
from lib.core.function import train, evaluation
from lib.core.post_process import final_result_process
from lib.core.utils_ab import weight_init
from lib.utils.utils import save_model, decay_lr, backup_codes, ModelEma


def parse_args():
    parser = argparse.ArgumentParser(description='SSAD temporal action localization')
    parser.add_argument('--cfg', type=str, help='experiment config file', default='./experiments/A2Net_thumos.yaml')
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
    # copy config file
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, 'code')
        backup_codes(cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LISTS)
    fix_random_seed(cfg.BASIC.SEED)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE

    # data loader
    train_dset = TALDataset(cfg, cfg.DATASET.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True, drop_last=False, num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)
    val_dset = TALDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False, drop_last=False, num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)

    model = LocNet(cfg)
    model.apply(weight_init)
    model.cuda()

    ema_model = ModelEma(model)

    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH+1):
        loss_train = train(cfg, train_loader, model, optimizer, ema_model, epoch)
        print('epoch %d: loss: %f' % (epoch, loss_train))
        with open(os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.LOG_FILE), 'a') as f:
            f.write("epoch %d, loss: %.4f\n" % (epoch, loss_train))

        # decay lr
        if epoch in cfg.TRAIN.LR_DECAY_EPOCHS:
            decay_lr(optimizer, factor=cfg.TRAIN.LR_DECAY_FACTOR)

        if epoch in cfg.TEST.EVAL_INTERVAL:
            save_model(cfg, epoch=epoch, model=model, optimizer=optimizer)


if __name__ == '__main__':
    main()

