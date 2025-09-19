
from tqdm import tqdm
from data import HCPDataFidelity
from nets import DCSARL, sh_to_sf_batch, data_fidelity
import torch
import argparse
import time
import os
import random
import numpy as np
from data import get_logger, batch_psnr_ssim, MyWFLoss
import wandb

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# plot
import matplotlib.pyplot as plt

# set environment
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
os.environ['WORLD_SIZE'] = '2'
os.environ['RANK'] = '1'  # rank
os.environ["WANDB_MODE"] = "disabled"


def train(rank, world_size):

    path = os.path.join('./checkpoint/', time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(path):
        os.mkdir(path)
    logger = get_logger(os.path.join(path, 'SHRL.log'))

    # -----------------------
    # parameters settings
    # -----------------------
    parser = argparse.ArgumentParser()

    # about SHRL model
    parser.add_argument('-encoder_name', type=str, default='RDN', dest='encoder_name',
                        help='the type of encoder network, including RDN .')
    parser.add_argument('-decoder_depth', type=int, default=8, dest='decoder_depth',
                        help='the depth of the decoder network (default=8).')
    parser.add_argument('-decoder_width', type=int, default=256, dest='decoder_width',
                        help='the width of the decoder network (default=256).')
    parser.add_argument('-in_dim', type=int, default=30, dest='in_dim',
                        help='the number of gradient.')
    parser.add_argument('-out_dim', type=int, default=30, dest='out_dim',
                        help='the width of the decoder network.')

    # about training and validation data
    parser.add_argument('-hr_data_train', type=str,
                        default='/media/sswang/5cbcc386-8cbf-4e32-b1ae-a704b2be3ffb/HCP/train_p3d',
                        dest='hr_data_train', help='the file path of HR patches for training')
    parser.add_argument('-hr_data_val', type=str,
                        default='/media/sswang/5cbcc386-8cbf-4e32-b1ae-a704b2be3ffb/HCP/val_p3d',
                        dest='hr_data_val', help='the file path of HR patches for validation')
    parser.add_argument('-bvecs', type=str,
                        default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/hcp'
                                '/DTI/3T/145x174x145/50_30/single-shell/b1000/bvecs',
                        dest='bvecs', help='the file path of bvecs for training')

    # about training hyper-parameters
    parser.add_argument('-lr', type=float, default=1e-4, dest='lr',
                        help='the initial learning rate')
    parser.add_argument('-epoch', type=int, default=200, dest='epoch',
                        help='the total number of epochs for training')
    parser.add_argument('-bs', type=int, default=5, dest='batch_size',
                        help='the number of LR-HR patch pairs')
    parser.add_argument('-seed', type=int, default=24, dest='seed')

    args = parser.parse_args()
    encoder_name = args.encoder_name
    decoder_depth = args.decoder_depth
    decoder_width = args.decoder_width
    in_dim = args.in_dim
    out_dim = args.out_dim
    hr_data_train = args.hr_data_train
    hr_data_val = args.hr_data_val
    bvecs = args.bvecs
    lr = args.lr
    epoch = args.epoch
    batch_size = args.batch_size
    seed = args.seed

    # -----------------------
    # display parameters
    # -----------------------
    logger.info('Parameter Settings')
    logger.info('')
    logger.info('------------File------------')
    logger.info('hr_data_train: {}'.format(hr_data_train))
    logger.info('hr_data_val: {}'.format(hr_data_val))
    logger.info('bvecs: {}'.format(bvecs))
    logger.info('------------Train-----------')
    logger.info('lr: {}'.format(lr))
    logger.info('batch_size_train: {}'.format(batch_size))
    logger.info('seed: {}'.format(seed))
    logger.info('gpu: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    logger.info('epochs: {}'.format(epoch))
    logger.info('------------Model-----------')
    logger.info('encoder_name : {}'.format(encoder_name))
    logger.info('decoder in_dim: {}'.format(in_dim))
    logger.info('decoder out_dim: {}'.format(out_dim))
    logger.info('decoder depth: {}'.format(decoder_depth))
    logger.info('decoder width: {}'.format(decoder_width))

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    layers = 3
    gamma = 1  # 1
    activation = 'ReLU'

    # wandb
    run = wandb.init(
        project="SHRL",
        config={
            "encoder_name": encoder_name,
            "decoder_depth": decoder_depth,
            "decoder_width": decoder_width,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "hr_data_train": hr_data_train,
            "hr_data_val": hr_data_val,
            "bvecs": bvecs,
            "lr": lr,
            "epoch": epoch,
            "batch_size": batch_size,
            'seed': seed,
            "gpu": os.environ['CUDA_VISIBLE_DEVICES'],
        },
        # mode='offline'
    )

    # initialize ditributed training
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # load data
    # -----------------------
    train_dataset = HCPDataFidelity(data_path=hr_data_train, bvecs_path=bvecs, g=in_dim, out=out_dim)
    val_dataset = HCPDataFidelity(data_path=hr_data_val, bvecs_path=bvecs, g=in_dim, out=out_dim)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=val_sampler)

    # -----------------------
    # model & optimizer DDP
    # -----------------------

    INRSSR = DCSARL(decoder_depth=int(decoder_depth / 2), decoder_width=decoder_width,
                    in_dim=in_dim, out_dim=out_dim, layers=layers, act=activation, device=rank).to(rank)
    INRSSR = torch.nn.parallel.DistributedDataParallel(INRSSR, device_ids=[rank])

    # define loss function and optimizer
    wf_func = MyWFLoss(gamma=gamma, device=rank)

    optimizer = torch.optim.AdamW(params=INRSSR.parameters(), lr=lr, weight_decay=1e-2)  # AdamW
    logger.info('optimizer: AdamW')

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

    n_parameters = sum(p.numel() for p in INRSSR.parameters() if p.requires_grad)
    logger.info('the parameter number of server_model is : {:.2f} M'.format(n_parameters / 1e6))

    # -----------------------
    # break out and load pre-trained model
    # -----------------------
    start_epoch = 0
    checkpoint_file = './checkpoint/20240916-215221/RDN_d30.pkl'
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location={'cuda:0': f'cuda:{rank}'})
        INRSSR.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])

        logger.info('load model from {}'.format('./checkpoint/20240916-215221/RDN_d30.pkl'))

    # -----------------------
    # training & validation
    # -----------------------
    best_model = {}
    best_psnr, best_ssim = 0., 0.
    train_loss, val_loss = [], []

    start_time = time.time()

    for ep in range(start_epoch, epoch):
        print('epoch:', ep+1)
        train_sampler.set_epoch(ep)

        INRSSR.train()
        loss_train, l1_train, wf_train = 0., 0., 0.
        train_psnr, train_ssim = [], []
        for i, (img_lr, xy_hr, img_hr, bvec, mask, index) in tqdm(enumerate(train_loader)):
            img_lr, xy_hr, img_hr = img_lr.to(rank), xy_hr.to(rank), img_hr.to(rank)
            bvec, mask = bvec.to(rank), mask.to(rank)

            # print(index.shape)
            N, C, H, W = img_hr.shape

            optimizer.zero_grad()
            img_pre = INRSSR(img_lr, xy_hr, bvec, mask, index[0])  

            # sh_to_sf
            pre = sh_to_sf_batch(img_pre, bvec, rank)  
            pre = pre.permute(0, 2, 1).reshape(N, C, H, W)  

            # dc
            pre = data_fidelity(img_lr[:, :, 1, ...], pre, mask[:, :, 1, ...], index[0])

            l1_loss, wf_loss, loss = wf_func(pre, img_hr)

            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            l1_train += l1_loss.item()
            wf_train += wf_loss.item()

            # metric
            psnr, ssim = batch_psnr_ssim(pre, img_hr)
            train_psnr.append(psnr)
            train_ssim.append(ssim)

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if rank == 0:
            logger.info('(TRAIN) Epoch[{}/{}], lr: {}, l1_loss: {:.6f}, wf_loss: {:.6f}, loss: {:.6f}, PSNR: {:.4f}, SSIM:'
                        '{:.4f}'.format(ep + 1, epoch, current_lr, l1_train / len(train_loader), wf_train /
                                        len(train_loader), loss_train / len(train_loader), np.mean(train_psnr),
                                        np.mean(train_ssim)))

            wandb.log({"loss_train": loss_train / len(train_loader)})
            wandb.log({"PSNR_train": np.mean(train_psnr)})
            wandb.log({"SSIM_train": np.mean(train_ssim)})

            train_loss.append(loss_train / len(train_loader))

        # validation
        val_sampler.set_epoch(ep)
        INRSSR.eval()
        loss_val, l1_val, wf_val = 0., 0., 0.
        val_psnr, val_ssim = [], []
        with torch.no_grad():
            for i, (img_lr, xy_hr, img_hr, bvec, mask, index) in tqdm(enumerate(val_loader)):
                img_lr, xy_hr, img_hr = img_lr.to(rank), xy_hr.to(rank), img_hr.to(rank)
                bvec, mask = bvec.to(rank), mask.to(rank)
                N, C, H, W = img_hr.shape

                img_pre = INRSSR(img_lr, xy_hr, bvec, mask, index[0])  

                # sh_to_sf
                pre = sh_to_sf_batch(img_pre, bvec, rank)  
                pre = pre.permute(0, 2, 1).reshape(N, C, H, W)  
                # print(pre.shape)
                # dc
                pre = data_fidelity(img_lr[:, :, 1, ...], pre, mask[:, :, 1, ...], index[0])

                l1_loss, wf_loss, loss = wf_func(pre, img_hr)

                loss_val += loss.item()
                l1_val += l1_loss.item()
                wf_val += wf_loss.item()

                psnr, ssim = batch_psnr_ssim(pre, img_hr)
                val_psnr.append(psnr)
                val_ssim.append(ssim)

        if best_psnr < np.mean(val_psnr) and best_ssim < np.mean(val_ssim):
            best_psnr = np.mean(val_psnr)
            best_ssim = np.mean(val_ssim)
            best_model = {
                'model': INRSSR.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': ep,
                'lr': current_lr
            }

        if rank == 0:
            # save model
            torch.save(best_model, os.path.join(path, '{}_d30.pkl'.format(encoder_name)))
            logger.info('current best epoch: {}'.format(best_model['epoch'] + 1))
            if ep+1 == 100:
                torch.save(best_model, os.path.join(path, '{}_d30_epoch100.pkl'.format(encoder_name)))

            logger.info('(VAL) Epoch[{}/{}], l1_loss: {:.6f}, wf_loss: {:.6f}, loss: {:.6f}, PSNR: {:.4f}, SSIM: '
                        '{:.4f}'.format(ep + 1, epoch, l1_val / len(val_loader), wf_val / len(val_loader),
                                       loss_val / len(val_loader), np.mean(val_psnr), np.mean(val_ssim)))

            wandb.log({"loss_val": loss_val / len(val_loader)})
            wandb.log({"PSNR_val": np.mean(val_psnr)})
            wandb.log({"SSIM_val": np.mean(val_ssim)})

            val_loss.append(loss_val / len(val_loader))

        # learning rate decays by half every some epochs.
        lr_scheduler.step()

    if rank == 0:
        logger.info('the best epoch :{}'.format(best_model['epoch']))
        logger.info('Training time: {} h'.format((time.time() - start_time) / 3600.))

        plt.figure()
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='val')
        plt.legend()
        plt.savefig(os.path.join(path, 'loss.png'))

    run.finish()
    logger.info('Finish!')

if __name__ == '__main__':
    world_size = int(os.environ['WORLD_SIZE'])  # 
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

