import argparse
import os
import time
from functools import partial
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

def SIAGT_attack(model, loss_fn, xn, x0, pred_query, coord, coord_adjacent, cell, alpha, num_iters, beta,is_transfer):
    inp_sub = torch.FloatTensor([0.5]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor([0.5]).view(1, -1, 1, 1).cuda()
    xn = (xn - inp_sub) / inp_div
    x0 = (x0 - inp_sub) / inp_div
    start_time = time.time()
    for i in range(num_iters):
        xn.requires_grad = True
        pred_attack = batched_predict_grad(model, xn, coord, cell, 30000)
        pred_attack_adjacent = batched_predict_grad(model, xn, coord_adjacent, cell, 30000)
        model.zero_grad()
        #if is_transfer:
        loss = loss_fn(pred_attack, pred_query) + beta * loss_fn(pred_attack_adjacent, pred_attack)
        #else:
            #loss = loss_fn(pred_attack, pred_query)
        loss.backward()
        images_grad = (alpha/(i+1)) * torch.sign(xn.grad.data)
        xn1 = torch.clamp(xn + images_grad, -1, 1)
        delta = torch.clamp(xn1 - x0, -alpha, alpha)
        xn = (x0 + delta).detach_()
    end_time = time.time()
    cost_time = end_time - start_time
    return xn, cost_time


def eval_psnr(loader, model, data_norm=None, setting=None):
    model.eval()
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

    pred_scales = setting['pred_scales']
    alpha = float(setting['alpha'])
    beta =  setting['beta']
    delta_g = float(setting['delta_g'])
    query_number = int(setting['query_number'])
    query_block = int(setting['query_block'])
    num_iters = int(setting['num_iters'])
    is_transfer = setting['is_transfer']
    PSNR = {}
    SSIM = {}
    LPIPS ={}
    for s in pred_scales:
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

    pbar = tqdm(loader, leave=False, desc='val')
    loss_fn = nn.MSELoss()
    for j, batch in enumerate(pbar):
        for k, v in batch.items():
            batch[k] = v.cuda()
        inp = (batch['inp'] - inp_sub) / inp_div
        ih, iw = batch['inp'].shape[-2:]
        ih, iw = round(ih / query_block), round(iw / query_block)
        feat_coord = utils.make_coord([ih, iw], flatten=True).cuda().unsqueeze(dim=0)
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
        coord = torch.cat(coord_list, dim=1)
        cell = batch['cell']
        coord_adjacent = coord.clone()
        deta = torch.empty_like(coord).uniform_(-delta_g,delta_g)
        coord_adjacent+=deta
        coord_adjacent.clamp_(-1, 1)
        pred_query = batched_predict(model, inp, coord, cell, 30000)
        perturbed_image = batch['inp'].clone()
        noise = torch.empty_like(perturbed_image).uniform_(-alpha, alpha)
        perturbed_image = torch.clamp(perturbed_image + noise, 0, 1)

        xn = perturbed_image
        x0 = batch['inp'].clone()
        images, time = SIAGT_attack(model, loss_fn, xn, x0, pred_query,coord, coord_adjacent, cell, alpha=alpha,
                                  num_iters=num_iters,beta = beta,is_transfer = is_transfer)
        save = images * gt_div + gt_sub
        save.clamp_(0, 1)
        save_images(save, j, tpye='lr')

        lr_ssim = utils.calculate_ssim(tensor2img(batch['inp'].squeeze()), tensor2img(save.squeeze()))
        lr_psnr = utils.calc_psnr(batch['inp'], save)
        lr_lpips = utils.calculate_lpips(batch['inp'], save)
        for s in pred_scales:
            shape = [round(ih * s), round(iw * s)]
            hr_coord = utils.make_coord(shape, flatten=True).cuda().unsqueeze(dim=0)
            cell = torch.ones_like(hr_coord)
            cell[:, :, 0] *= 2 / shape[0]
            cell[:, :, 1] *= 2 / shape[1]

            pred_attack = batched_predict(model, images, hr_coord, cell, 30000)
            pred_attack = pred_attack * gt_div + gt_sub
            pred_attack.clamp_(0, 1)

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


    print('LR: ')
    print('LR_PSNR: {:.4f}'.format(LR_psnr.item()))
    print('LR_SSIM: {:.4f}'.format(LR_ssim.item()))
    print('Cost_time: {:.4f}'.format(Cost_time.item()))
    print('LR_LPIPS: {:.4f}'.format(LR_lpips.item()))

    for s in pred_scales:
        print('scale: ' + str(s))
        print('SR_PSNR: {:.4f}'.format(PSNR[s].item()))
        print('SR_SSIM: {:.4f}'.format(SSIM[s].item()))
        #print('SR_LPIPS: {:.4f}'.format(LPIPS[s].item().item()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/test_attack/test-liif-attack.yaml')
    parser.add_argument('--model', default='pre-models/edsr-baseline-liif.pth') # or pre-models/edsr-baseline-lte.pth
    parser.add_argument('--gpu', default='0')
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
    setting = config['attack_setting']

    eval_psnr(loader, model,
              data_norm=config.get('data_norm'),
              setting=setting)
