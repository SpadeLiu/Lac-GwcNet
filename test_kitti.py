import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import os
from tqdm import tqdm, trange
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from dataloader import KITTIloader as kt
from networks.stackhourglass import PSMNet
import loss_functions as lf


parser = argparse.ArgumentParser(description='LaC')
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='2')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data_path', type=str, default='/media/data/dataset/KITTI/data_scene_flow/training/')
parser.add_argument('--load_path', type=str, default='state_dicts/kitti2015.pth')
parser.add_argument('--max_disp', type=int, default=192)
parser.add_argument('--lsp_width', type=int, default=3)
parser.add_argument('--lsp_height', type=int, default=3)
parser.add_argument('--lsp_dilation', type=list, default=[1, 2, 4, 8])
parser.add_argument('--lsp_mode', type=str, default='separate')
parser.add_argument('--lsp_channel', type=int, default=4)
parser.add_argument('--no_udc', action='store_true', default=False)
parser.add_argument('--refine', type=str, default='csr')
args = parser.parse_args()

if not args.no_cuda:
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)


all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt.kt_loader(args.data_path)

affinity_settings = {}
affinity_settings['win_w'] = args.lsp_width
affinity_settings['win_h'] = args.lsp_width
affinity_settings['dilation'] = args.lsp_dilation
udc = not args.no_udc

model = PSMNet(maxdisp=args.max_disp, struct_fea_c=args.lsp_channel, fuse_mode=args.lsp_mode,
               affinity_settings=affinity_settings, udc=udc, refine=args.refine)
model = nn.DataParallel(model)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
if cuda:
    model.cuda()
model.eval()

ckpt = torch.load(args.load_path)
model.load_state_dict(ckpt)

mae = 0
op = 0
for i in trange(len(test_limg)):
    limg = Image.open(test_limg[i]).convert('RGB')
    rimg = Image.open(test_rimg[i]).convert('RGB')

    w, h = limg.size
    limg = limg.crop((w - 1232, h - 368, w, h))
    rimg = rimg.crop((w - 1232, h - 368, w, h))

    limg_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
    rimg_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)
    limg_tensor = limg_tensor.unsqueeze(0).cuda()
    rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

    disp_gt = Image.open(test_ldisp[i])
    disp_gt = disp_gt.crop((w - 1232, h - 368, w, h))
    disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32) / 256
    gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        pred_disp = model(limg_tensor, rimg_tensor, gt_tensor)

    predict_np = pred_disp.squeeze().cpu().numpy()

    op_thresh = 3
    mask = (disp_gt > 0)
    error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

    op += np.sum((error > op_thresh) & (error > disp_gt * 0.05)) / np.sum(mask)
    mae += np.mean(error[mask])

print('OP: %.2f%%' % (op / len(test_limg) * 100))
print('MAE: %.3f' % (mae / len(test_limg)))