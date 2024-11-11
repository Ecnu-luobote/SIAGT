import argparse
import os
import math
import time
from functools import partial

import torch.nn.functional as F
import numpy as np
import yaml
import torch
from basicsr import tensor2img
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from torchvision import transforms
from torch.autograd import Variable


def save_images(img, index, tpye='lr'):
    img = img.squeeze().cpu()
    save_path = 'save/' + str(index) + tpye + '.png'

    transforms.ToPILImage()(img).save(save_path)


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def batched_predict_grad(model, inp, coord, cell, bsize):
    model.gen_feat(inp)
    n = coord.shape[1]
    ql = 0
    preds = []

    while ql < n:
        qr = min(ql + bsize, n)
        pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
        preds.append(pred)
        ql = qr
    pred = torch.cat(preds, dim=1)
    return pred


# images 初始化的攻击图像,inp原始图像x0,pred f(x0)
def pgd_attack(model, loss_fn, xn, x0, pred1, pred2, coord1, coord2, cell, alpha, num_iters):
    inp_sub = torch.FloatTensor([0.5]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor([0.5]).view(1, -1, 1, 1).cuda()
    xn = (xn - inp_sub) / inp_div
    x0 = (x0 - inp_sub) / inp_div
    Loss = []
    a = time.time()
    for i in range(num_iters):
        xn.requires_grad = True
        pred_attack1 = batched_predict_grad(model, xn, coord1, cell, 30000)
        pred_attack2 = batched_predict_grad(model, xn, coord2, cell, 30000)
        model.zero_grad()
        loss = loss_fn(pred_attack1, pred1) + 2 * loss_fn(pred_attack2, pred_attack1)
        Loss.append(loss)
        # print(loss)
        loss.backward()
        # 扰动项
        images_grad = (alpha/(i+1)) * torch.sign(xn.grad.data)
        # 添加扰动 X_n+ images_grad
        xn1 = torch.clamp(xn + images_grad, -1, 1)
        # 限制扰动范围
        delta = torch.clamp(xn1 - x0, -alpha, alpha)
        # 添加扰动
        xn = (x0 + delta).detach_()
    a1 = time.time()
    cost_time = a1 - a
    return xn, cost_time


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, setting=None,
              verbose=False):
    model.eval()

    # 指标函数与数据预处理
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    scale = [18,24,30]
    PSNR = {}
    SSIM = {}
    LPIPS ={}
    for s in scale:
        SR_psnr = utils.Averager()
        SR_ssim = utils.Averager()
        SR_lpips = utils.Averager()
        PSNR[s] = SR_psnr
        SSIM[s] = SR_ssim
        LPIPS[s] = SR_lpips
    LR_psnr = utils.Averager()
    LR_ssim = utils.Averager()
    LR_lpips = utils.Averager()

    Cost_time = utils.Averager()
    alpha = setting['alpha']
    print(alpha)
    num_iters = setting['num_iters']
    pbar = tqdm(loader, leave=False, desc='val')

    # 损失函数
    loss_fn = nn.MSELoss()
    query_number = 4
    for j, batch in enumerate(pbar):
        for k, v in batch.items():
            batch[k] = v.cuda()
        # 原图
        inp = (batch['inp'] - inp_sub) / inp_div
        ih, iw = batch['inp'].shape[-2:]
        feat_coord = utils.make_coord(inp.shape[-2:], flatten=True).cuda().unsqueeze(dim=0)
        rx = 2 / batch['inp'].shape[-2] / 2
        ry = 2 / batch['inp'].shape[-1] / 2
        coord_list = []
        for i in range(query_number):
            deta_x = torch.empty_like(feat_coord[:, :, 0]).uniform_(-rx, rx)
            deta_y = torch.empty_like(feat_coord[:, :, 1]).uniform_(-ry, ry)
            coord = feat_coord.clone()
            coord[:, :, 0] += deta_x
            coord[:, :, 1] += deta_y
            coord.clamp_(-1, 1)
            coord_list.append(coord)
            # 取点的数量
        coord = torch.cat(coord_list, dim=1)
        cell = batch['cell']
        coord_2 = coord.clone()
        deta = torch.empty_like(coord).uniform_(-0.005, 0.005)
        coord_2+=deta
        coord_2.clamp_(-1, 1)
        pred1 = batched_predict(model, inp, coord, cell, 30000)
        pred2 = batched_predict(model, inp, coord_2, cell, 30000)
        # 初始化输入图像
        perturbed_image = batch['inp'].clone()
        noise = torch.empty_like(perturbed_image).uniform_(-alpha, alpha)
        perturbed_image = torch.clamp(perturbed_image + noise, 0, 1)
        # 攻击
        xn = perturbed_image
        x0 = batch['inp'].clone()
        images, time = pgd_attack(model, loss_fn, xn, x0, pred1, pred2, coord, coord_2, cell, alpha=alpha,
                                  num_iters=num_iters)
        save = images * gt_div + gt_sub
        save.clamp_(0, 1)
        # 保存攻击后的图像
        save_images(save, j, tpye='lr')
        lr_ssim = utils.calculate_ssim(tensor2img(batch['inp'].squeeze()), tensor2img(save.squeeze()))
        lr_psnr = utils.calc_psnr(batch['inp'], save)
        lr_lpips = utils.calculate_lpips(batch['inp'], save)
        #model_lte = models.make(torch.load('pre-models/edsr-baseline-lte.pth')['model'], load_sd=True).cuda()
        for s in scale:
            shape = [round(ih * s), round(iw * s)]
            hr_coord = utils.make_coord(shape, flatten=True).cuda().unsqueeze(dim=0)
            cell = torch.ones_like(hr_coord)
            cell[:, :, 0] *= 2 / shape[0]
            cell[:, :, 1] *= 2 / shape[1]
            # 预测攻击图像的SR
            pred_attack = batched_predict(model, images, hr_coord, cell, 30000)
            pred_attack = pred_attack * gt_div + gt_sub
            pred_attack.clamp_(0, 1)
            # 预测原始图像的SR
            pred = batched_predict(model, inp, hr_coord, cell, 30000)
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred_attack = pred_attack.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            save_images(pred_attack, j, tpye='sr')

            metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=s)
            # 攻击后图像的预测值与原始图像预测值的PSNR
            sr_psnr = metric_fn(pred_attack, pred)
            sr_ssim = utils.calculate_ssim(tensor2img(pred_attack.squeeze()), tensor2img(pred.squeeze()), border=s)
            sr_lpips = utils.calculate_lpips(pred_attack, pred)
            PSNR[s].add(sr_psnr.item(), inp.shape[0])
            SSIM[s].add(sr_ssim, inp.shape[0])
            LPIPS[s].add(sr_lpips.item(),inp.shape[0])


        LR_psnr.add(lr_psnr.item(), inp.shape[0])
        LR_ssim.add(lr_ssim, inp.shape[0])
        LR_lpips.add(lr_lpips.item(), inp.shape[0])
        Cost_time.add(time, inp.shape[0])

    for s in scale:
        print(str(s))
        print('SR_PSNR: {:.4f}'.format(PSNR[s].item()))
        print('SR_SSIM: {:.4f}'.format(SSIM[s].item()))
        print('SR_LPIPS: {:.4f}'.format(LPIPS[s].item()))

    print('LR_PSNR: {:.4f}'.format(LR_psnr.item()))
    print('LR_SSIM: {:.4f}'.format(LR_ssim.item()))
    print('Cost_time: {:.4f}'.format(Cost_time.item()))
    #print(LR_lpips.c)
    #print('LR_LPIPS: {:.4f}'.format(LR_lpips.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/test_attack/test-b100-2.yaml')
    parser.add_argument('--model', default='pre-models/edsr-baseline-liif.pth')
    parser.add_argument('--gpu', default='4')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    setting = {}
    setting['alpha'] = 8 / 255.
    setting['num_iters'] = 50
    setting['scale'] = config.get('scale')

    eval_psnr(loader, model,
              data_norm=config.get('data_norm'),
              eval_type=config.get('eval_type'),
              eval_bsize=config.get('eval_bsize'),
              setting=setting,
              verbose=True)
