import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm, trange
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import cv2

from dataloader import sceneflow_loader as sf
from dataloader import readpfm as rp
from networks.stackhourglass import PSMNet
import loss_functions as lf


parser = argparse.ArgumentParser(description='LaC')
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='2')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data_path', type=str, default='/media/data/dataset/SceneFlow/')
parser.add_argument('--load_path', type=str, default='state_dicts/SceneFlow.pth')
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

all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = sf.sf_loader(args.data_path)

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
    limg_path = test_limg[i]
    rimg_path = test_rimg[i]

    limg = Image.open(limg_path).convert('RGB')
    rimg = Image.open(rimg_path).convert('RGB')

    # rimg = lf.random_noise(rimg, type='illumination')
    # rimg = lf.random_noise(rimg, type='color')
    # rimg = lf.random_noise(rimg, type='haze')
    #
    w, h = limg.size
    limg = limg.crop((w - 960, h - 544, w, h))
    rimg = rimg.crop((w - 960, h - 544, w, h))

    limg_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
    rimg_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)
    limg_tensor = limg_tensor.unsqueeze(0).cuda()
    rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

    disp_gt, _ = rp.readPFM(test_ldisp[i])
    disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
    gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        pred_disps = model(limg_tensor, rimg_tensor, gt_tensor)
        pred_disps = pred_disps[:, 4:, :]

    predict_np = pred_disps.squeeze().cpu().numpy()

    mask = (disp_gt < args.max_disp) & (disp_gt > 0)
    if len(disp_gt[mask]) == 0:
        continue

    op_thresh = 1

    error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
    op += np.sum(error > op_thresh) / np.sum(mask)
    mae += np.mean(error[mask])

print(mae / len(test_limg))
print(op / len(test_limg))
