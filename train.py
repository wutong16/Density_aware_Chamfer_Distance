import torch
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse
import shutil

import torch.optim as optim
from torch.optim import Adam
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from data.mvp_new import MVP_CP
from utils.train_utils import *
from utils.env import init_dist, get_root_logger, set_random_seed

   
def train():
    logger.info(str(args))
    if args.eval_emd:
        metrics = ['dcd', 'cd_t', 'emd', 'f1']
    else:
        metrics = ['dcd', 'cd_t', 'f1']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    # dataset
    dataset = MVP_CP(prefix="train")
    dataset_test = MVP_CP(prefix="val")
    cat_name = dataset_test.cat_name
    cat_number = len(cat_name)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                                            shuffle=shuffle, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, sampler=test_sampler,
                                            shuffle=True, num_workers=int(args.workers))
    logger.info('Length of train dataset:%d', len(dataset))
    logger.info('Length of test dataset:%d', len(dataset_test))

    # random seed
    seed = int(args.manual_seed) if args.manual_seed else random.randint(1, 10000)
    logger.info('Random Seed: %d' % seed)
    set_random_seed(seed)

    # model
    shutil.copyfile('models/%s.py' % args.model_name, os.path.join(log_dir,'%s.py' % args.model_name))
    shutil.copyfile(arg.config, os.path.join(log_dir,arg.config.split("/")[-1]))
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = model_module.Model(args).cuda()
    if args.distributed:
        net = DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,
                                      find_unused_parameters=True)
    else:
        net = DataParallel(net)
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)

    # learning rate
    lr = args.lr
    decay_epoch_list, decay_rate_list = [], []
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    # optimizer
    optimizer, sub_optimizers, sub_lrs = get_optimizer(net, args, lr)

    alpha = None
    varying_constant_epochs, varying_constant = [], []
    if args.varying_constant:
        varying_constant_epochs = [int(ep.strip()) for ep in args.varying_constant_epochs.split(',')]
        varying_constant = [float(c.strip()) for c in args.varying_constant.split(',')]
        assert len(varying_constant) == len(varying_constant_epochs) + 1

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        logger.info("%s's previous weights loaded from %s" % (args.model_name, args.load_model))

    if args.test_only:
        val(net, 0, val_loss_meters, dataloader_test, eval_cat=True, cat_name=cat_name,
        cat_number=cat_number, metrics=metrics, best_epoch_losses=None, test_only=True)
        exit()

    val(net, 0, val_loss_meters, dataloader_test, best_epoch_losses, simple_eval=True)

    for epoch in range(int(args.start_epoch), args.nepoch):
        train_loss_meter.reset()
        net.module.train()
        # varying alpha
        if args.varying_constant:
            for ind, ep in enumerate(varying_constant_epochs):
                if epoch < ep:
                    alpha = varying_constant[ind]
                    break
                elif ind == len(varying_constant_epochs)-1 and epoch >= ep:
                    alpha = varying_constant[ind+1]
                    break
        # lr decay
        if args.lr_decay:
            decay = 1
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    decay = decay * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    decay = decay * decay_rate_list[decay_epoch_list.index(epoch)]
            lr = lr * decay
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            for op, sub_lr in zip(sub_optimizers, sub_lrs):
                for param_group in op.param_groups:
                    param_group['lr'] = sub_lr * decay
        # train
        for i, data in enumerate(dataloader, 0):

            optimizer.zero_grad()
            for op in sub_optimizers:
                op.zero_grad()

            _, inputs, gt = data

            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()

            out2, loss2, net_loss, losses = net(inputs, gt, alpha=alpha)

            train_loss_meter.update(net_loss.mean().item())
            net_loss = net_loss.mean()

            net_loss.backward()

            optimizer.step()
            for op in sub_optimizers:
                op.step()

            if i % args.step_interval_to_print == 0 and local_rank == 0:
                extra_loss_info = ''
                if losses is not None:
                    for k,v in losses.items():
                        extra_loss_info += "%s: %f" % (k, v.mean().item())
                logger.info(exp_name + ' train [%d: %d/%d]  loss_type: %s, fine_loss: %f total_loss: %f %s lr: %f' % (
                    epoch, i, len(dataset) / args.batch_size, args.loss, loss2.mean().item(), net_loss.mean().item(),
                 extra_loss_info, lr) + ' alpha: ' + str(alpha))

        if epoch % args.epoch_interval_to_save == 0 and local_rank == 0:
            save_model('%s/network.pth' % log_dir, net, net_d=None)
            logger.info("Saving net...")
        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses,
                eval_cat=False, cat_name=cat_name, cat_number=cat_number, metrics=metrics)

def val(net, curr_epoch_num, val_loss_meters, dataloader_test,best_epoch_losses=None,
        test_only=False, simple_eval=False, test_emd=True,
        eval_cat=False, cat_number=16, cat_name=None, metrics=None):
    if simple_eval:
        logging.info('Making sure that the test step runs well.')
    else:
        logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()
    test_loss_cat = torch.zeros([cat_number, len(metrics)], dtype=torch.float32).cuda() if eval_cat else None
    test_cat_num = torch.zeros([cat_number])
    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            label, inputs, gt = data
            curr_batch_size = gt.shape[0]
            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()
            result_dict = net(inputs, gt, is_training=False, test_emd=test_emd)
            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item(), curr_batch_size)
            if simple_eval:
                return
            if eval_cat:
                for j, l in enumerate(label):
                    for ind, m in enumerate(metrics):
                        if not m in result_dict:
                            continue
                        test_loss_cat[int(l), ind] += result_dict[m][int(j)]
                    test_cat_num[int(l)] += 1
        if eval_cat:
            logger.info('Loss per category:')
            category_log = ''
            scalars = dict(cd_t=10000, emd=100, dcd=1, f1=100)
            # print(test_cat_num)
            for i in range(cat_number):
                category_log += 'category name: %s | ' % (cat_name[i])
                for ind, m in enumerate(metrics):
                    scale_factor = scalars[m]
                    category_log += '%s: %.3f |' % (m, test_loss_cat[i, ind] / test_cat_num[i] * scale_factor)
                category_log += '\n'
            logger.info(category_log)
        if test_only:
            curr_log = ''
            for loss_type, meter in val_loss_meters.items():
                curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)
            logger.info(curr_log)
            return
        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        #save_model('%s/epoch_%s_network.pth' % (log_dir, loss_type), net)
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                logger.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logger.info(curr_log)
        logger.info(best_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--test_only', help='whether to test on val set', action='store_true')

    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    time = datetime.datetime.now().isoformat()[:19]
    args.test_only = arg.test_only

    if arg.launcher == 'none':
        args.distributed = False
        local_rank = 0
    else:
        args.distributed = True
        init_dist(arg.launcher)
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    print("Using", torch.cuda.device_count(), "GPUs.")

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model and not args.flag:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name + '_' + args.loss + '_' + args.flag + '_' + time
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if args.start_epoch > 0:
            print("reset start epoch to be zero.")
            args.start_epoch = 0
    print("Exp name: ", exp_name)

    logger = get_root_logger(logging.INFO, handlers=[
        logging.FileHandler(os.path.join(log_dir, 'train.log'))])

    train()



