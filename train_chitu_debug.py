
import os
import time
import argparse
import os.path as osp
import numpy as np
import torch
import torch.distributed as dist

from utils.metric_util import MeanIoU
from utils.load_save_util import revise_ckpt, revise_ckpt_2
from dataloader.dataset import get_nuScenes_label_name
from builder import loss_builder, data_builder_chitu

import mmcv
from mmcv import Config
from mmcv.runner import build_optimizer
from mmseg.utils import get_root_logger
from timm.scheduler import CosineLRScheduler

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass


def main(args):
    # global settings
    torch.backends.cudnn.benchmark = True
    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    dataset_config = cfg.dataset_params
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader

    max_num_epochs = cfg.max_epochs
    grid_size = cfg.grid_size

    os.makedirs(args.work_dir, exist_ok=True)
    cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    my_model = model_builder.build(cfg.model)
    n_parameters = sum(p.numel()
                       for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')

    my_model = my_model.cuda()
    print('done model')
    
    unique_label_str = ['person', 'bicycle', 'car', 'bus', 'truck', 'forklift',
                        'trailer', 'rack', 'shelves', 'traffic_cone', 'goods', 'traffic_light', 'other_vehicle']
    train_dataset_loader, val_dataset_loader = \
        data_builder_chitu.build(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=grid_size,
            dist=False,
            scale_rate=cfg.get('scale_rate', 1)
        )

    # get optimizer, loss, scheduler
    unique_label = cfg.unique_label
    ignore_label = cfg.ignore_label
    optimizer = build_optimizer(my_model, cfg.optimizer)
    loss_func, lovasz_softmax = \
        loss_builder.build(ignore_label=ignore_label)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader)*max_num_epochs,
        lr_min=1e-6,
        warmup_t=500,
        warmup_lr_init=1e-5,
        t_in_epochs=False
    )
    
    CalMeanIou_vox = MeanIoU(
        unique_label, ignore_label, unique_label_str, 'vox')
    CalMeanIou_pts = MeanIoU(
        unique_label, ignore_label, unique_label_str, 'pts')

    # resume and load
    epoch = 0
    best_val_miou_pts, best_val_miou_vox = 0, 0
    global_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    print('resume from: ', cfg.resume_from)
    print('work dir: ', args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(
            revise_ckpt(ckpt['state_dict']), strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_miou_pts' in ckpt:
            best_val_miou_pts = ckpt['best_val_miou_pts']
        if 'best_val_miou_vox' in ckpt:
            best_val_miou_vox = ckpt['best_val_miou_vox']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))
    # training
    print_freq = cfg.print_freq

    while epoch < max_num_epochs:
        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, (imgs, img_metas, train_vox_label, train_grid, train_pt_labs) in enumerate(train_dataset_loader):
            '''
            imgs: torch.Size([1, 6, 3, 928, 1600])
            len(img_metas)=1, img_metas[0].keys():dict_keys(['lidar2img', 'img_shape']) 
            train_vox_label: torch.Size([1, 100, 100, 8])
            train_grid: torch.Size([1, 34752, 3])
            train_pt_labs: torch.Size([1, 34752, 1])
            '''
            imgs = imgs.cuda()
            train_grid = train_grid.to(torch.float32).cuda()
            if cfg.lovasz_input == 'voxel' or cfg.ce_input == 'voxel':
                voxel_label = train_vox_label.type(torch.LongTensor).cuda()
            if cfg.lovasz_input == 'points' or cfg.ce_input == 'points':
                train_pt_labs = train_pt_labs.cuda()
            # forward + backward + optimize
            data_time_e = time.time()
            outputs_vox, outputs_pts = my_model(
                img=imgs, img_metas=img_metas, points=train_grid)
            '''
            outputs_vox: torch.Size([1, 18, 100, 100, 8])
            outputs_pts: torch.Size([1, 18, 34752, 1, 1])
            '''
            if cfg.lovasz_input == 'voxel':
                lovasz_input = outputs_vox
                lovasz_label = voxel_label  # torch.Size([1, 100, 100, 8])
            else:
                lovasz_input = outputs_pts
                lovasz_label = train_pt_labs
            if cfg.ce_input == 'voxel':
                ce_input = outputs_vox
                ce_label = voxel_label
            else:
                ce_input = outputs_pts.squeeze(-1).squeeze(-1)
                ce_label = train_pt_labs.squeeze(-1)

            loss = lovasz_softmax(
                torch.nn.functional.softmax(lovasz_input, dim=1),
                lovasz_label, ignore=ignore_label
            ) + loss_func(ce_input, ce_label)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                my_model.parameters(), cfg.grad_max_norm)
            optimizer.step()
            loss_list.append(loss.item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.1f, lr: %.7f, time: %.3f (%.3f)' % (
                    epoch, i_iter, len(train_dataset_loader),
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s
                ))
            data_time_s = time.time()
            time_s = time.time()

        # save checkpoint
       
        dict_to_save = {
            'state_dict': my_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'global_iter': global_iter,
            'best_val_miou_pts': best_val_miou_pts,
            'best_val_miou_vox': best_val_miou_vox
        }
        save_file_name = os.path.join(os.path.abspath(
            args.work_dir), f'epoch_{epoch+1}.pth')
        torch.save(dict_to_save, save_file_name)
        dst_file = osp.join(args.work_dir, 'latest.pth')
        mmcv.symlink(save_file_name, dst_file)

        epoch += 1

        # eval
        my_model.eval()
        val_loss_list = []
        CalMeanIou_pts.reset()
        CalMeanIou_vox.reset()

        with torch.no_grad():
            for i_iter_val, (imgs, img_metas, val_vox_label, val_grid, val_pt_labs) in enumerate(val_dataset_loader):
                imgs = imgs.cuda()
                val_grid_float = val_grid.to(torch.float32).cuda()
                val_grid_int = val_grid.to(torch.long).cuda()
                vox_label = val_vox_label.cuda()
                val_pt_labs = val_pt_labs.cuda()

                predict_labels_vox, predict_labels_pts = my_model(
                    img=imgs, img_metas=img_metas, points=val_grid_float)
                if cfg.lovasz_input == 'voxel':
                    lovasz_input = predict_labels_vox
                    lovasz_label = vox_label
                else:
                    lovasz_input = predict_labels_pts
                    lovasz_label = val_pt_labs

                if cfg.ce_input == 'voxel':
                    ce_input = predict_labels_vox
                    ce_label = vox_label
                else:
                    ce_input = predict_labels_pts.squeeze(-1).squeeze(-1)
                    ce_label = val_pt_labs.squeeze(-1)

                loss = lovasz_softmax(
                    torch.nn.functional.softmax(lovasz_input, dim=1).detach(),
                    lovasz_label, ignore=ignore_label
                ) + loss_func(ce_input.detach(), ce_label)

                predict_labels_pts = predict_labels_pts.squeeze(-1).squeeze(-1)
                predict_labels_pts = torch.argmax(
                    predict_labels_pts, dim=1)  # bs, n
                predict_labels_pts = predict_labels_pts.detach().cpu()
                val_pt_labs = val_pt_labs.squeeze(-1).cpu()

                predict_labels_vox = torch.argmax(predict_labels_vox, dim=1)
                predict_labels_vox = predict_labels_vox.detach().cpu()
                val_grid_int = val_grid_int.detach().cpu()
                for count in range(len(val_grid_int)):
                    CalMeanIou_pts._after_step(
                        predict_labels_pts[count], val_pt_labs[count])
                    CalMeanIou_vox._after_step(
                        predict_labels_vox[
                            count,
                            val_grid_int[count][:, 0],
                            val_grid_int[count][:, 1],
                            val_grid_int[count][:, 2]].flatten(),
                        val_pt_labs[count])
                val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val % print_freq == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)' % (
                        epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))

        val_miou_pts = CalMeanIou_pts._after_epoch()
        val_miou_vox = CalMeanIou_vox._after_epoch()

        if best_val_miou_pts < val_miou_pts:
            best_val_miou_pts = val_miou_pts
        if best_val_miou_vox < val_miou_vox:
            best_val_miou_vox = val_miou_vox

        logger.info('Current val miou pts is %.3f while the best val miou pts is %.3f' %
                    (val_miou_pts, best_val_miou_pts))
        logger.info('Current val miou vox is %.3f while the best val miou vox is %.3f' %
                    (val_miou_vox, best_val_miou_vox))
        logger.info('Current val loss is %.3f' %
                    (np.mean(val_loss_list)))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--py-config', default='config/tpv04_occupancy_chitu.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_occupancy_chitu')
    parser.add_argument('--resume-from', type=str, default='')

    args = parser.parse_args()

    args.gpus = 1
    print(args)
    main(args)

# CUDA_VISIBLE_DEVICES=0 python train_debug.py --py-config config/tpv04_occupancy.py --work-dir out/tpv_occupancy
