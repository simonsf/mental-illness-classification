import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.python.logging_helpers import setup_logger
from utils.python.file_tools import load_config
from utils.train_tools import EpochConcateSampler
from models.network import *
from models.AttLoss import *


def train(config_file):
    assert torch.cuda.is_available(), 'CUDA is not available! Please check nvidia driver!'
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    cfg = load_config(config_file)
    cfg = cfg.cfg
    root_dir = os.path.dirname(config_file)
    cfg.general.save_dir = os.path.join(root_dir, cfg.general.save_dir)

    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    torch.cuda.manual_seed(cfg.general.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    val_flag = cfg.general.validation_list is not None

    train_dataset = ClassificationDataset(
        imlist_file=cfg.general.train_list,
        num_classes=cfg.dataset.num_classes,
        spacing=cfg.dataset.spacing,
        crop_size=cfg.dataset.crop_size,
        bag_size=cfg.dataset.bag_size,
        default_values=cfg.dataset.default_values,
        random_translation=cfg.dataset.random_translation,
        interpolation=cfg.dataset.interpolation,
        crop_normalizers=cfg.dataset.crop_normalizers
    )
  

    index_dict = class_index(train_dataset)
    sampler = EpochConcateSampler_balance(cfg.train.epochs, index_dict)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.train.batchsize,
                                  sampler=sampler,
                                  num_workers=cfg.train.num_threads,
                                  pin_memory=True
                                 )

    net = MILinMIL_Cls(
        img_ch=train_dataset.num_modality(),
        out_cls_ch=cfg.dataset.num_classes,
        bag_size=cfg.dataset.bag_size)

    log_file = os.path.join(cfg.general.save_dir, 'train_log.txt')
    logger = setup_logger(log_file, cfg.net.name + ' train')

    gpu_ids = list(range(cfg.general.num_gpus))
    net = nn.parallel.DataParallel(net, device_ids=gpu_ids)
    net = net.cuda()

    opt = optim.SGD(net.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)

    if cfg.general.resume_epoch >= 0:
        net = torch.load(os.pth.join(cfg.general.save_dir, 'checkpoints', 'chk_{}'.cfg.general.resume_epoch))
        last_save_epoch, batch_start = cfg.general.resume_epoch, 0
    else:
        last_save_epoch, batch_start = 0, 0
    batch_idx = batch_start
    data_iter = iter(train_dataloader)

    learning_rate = cfg.train.lr
    lr_now = learning_rate

    cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    att_loss_fn = TotalWeightedMSE(num_class=cfg.dataset.num_classes)

    last_record_epoch = last_save_epoch
    for i in range(len(train_dataloader)):
        net = net.train()
        data = data_iter.next()
        t1_imgs = data['t1'].cuda().float()
        inp_t1 = torch.cat([t1_imgs[i] for i in range(t1_imgs.shape[0])], dim=0)
        if 't2' in data:
            t2_imgs = data['t2'].cuda().float()
            inp_t2 = torch.cat([t2_imgs[i] for i in range(t2_imgs.shape[0])], dim=0)
            inp = torch.cat([inp_t1, inp_t2], dim=1)
        else:
            inp = inp_t1
        masks = data['mask'].cuda().float()
        #print('imgs: ', imgs.shape)

        begin_t = time.time()
        cls_labs = torch.tensor(np.array(data['label']).astype(np.int8)).cuda().long().unsqueeze(1)
        mask = torch.cat([masks[i] for i in range(t1_imgs.shape[0])], dim=0)

        opt.zero_grad()

        cls_pred, att, _ = net(inp)

        cls_loss = cls_loss_fn(cls_pred, cls_labs)
        att_loss = att_loss_fn(att.unsqueeze(2), mask.unsqueeze(2))

        loss = cls_loss + cfg.loss.cam_weight * att_loss
        loss.backward()
        opt.step()

        epoch_idx = batch_idx * cfg.train.batchsize // sampler.data_length

        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / cfg.train.batchsize

        if (batch_idx + 1) % cfg.train.plot_snapshot == 0:
            train_loss_plot_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
            plot_loss(log_file, train_loss_plot_file, name='train_loss',
                      display='Training Loss')

        if epoch_idx != last_record_epoch:
            msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, cls_loss: {:.4f}, ' \
                  'cam_loss: {:.4f}, learning_rate:{:.4g}, time: {:.4f} s/vol'
            msg = msg.format(epoch_idx, batch_idx, loss.item(), cls_loss.item(), att_loss.item(), lr_now,
                             sample_duration)
            logger.info(msg)
            last_record_epoch = epoch_idx

        if epoch_idx != 0 and (epoch_idx % cfg.train.save_epochs == 0):
            if last_save_epoch != epoch_idx:
                torch.save(net, os.path.join(cfg.general.save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx)))
                last_save_epoch = epoch_idx


def main():

    long_description = "Mutiple instance learning model for brain MR images"

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', nargs='?', default="/home/yichu/brain_classification/config.py",
                        help='volumetric segmentation2d train config file')
    args = parser.parse_args()
    train(args.input)


if __name__ == '__main__':
    main()












