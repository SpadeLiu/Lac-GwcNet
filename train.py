import argparse
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
from tqdm import tqdm

from dataloader import sceneflow_loader as sf
from networks.stackhourglass import PSMNet

parser = argparse.ArgumentParser(description='LaC')
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='0, 1')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--data_path', type=str, default='/media/data/dataset/SceneFlow/')
parser.add_argument('--save_path', type=str, default='trained_SceneFlow/')
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

trainLoader = torch.utils.data.DataLoader(
    sf.myDataset(all_limg, all_rimg, all_ldisp, training=True),
    batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)

testLoader = torch.utils.data.DataLoader(
    sf.myDataset(test_limg, test_rimg, test_ldisp, training=False),
    batch_size=1, shuffle=False, num_workers=2, drop_last=False)

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

optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))


def train(imgL, imgR, disp_true):
    model.train()
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    disp_true = torch.FloatTensor(disp_true)

    if cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    optimizer.zero_grad()

    loss1, loss2 = model(imgL, imgR, disp_true)
    loss1 = torch.mean(loss1)
    loss2 = torch.mean(loss2)

    if udc:
        loss = 0.1 * loss1 + loss2
    else:
        loss = loss1

    loss.backward()
    optimizer.step()

    return loss.item()


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    with torch.no_grad():
       output = model(imgL, imgR)
       output = output[:, 4:, :]

    mask = disp_true < args.maxdisp
    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))

    return loss


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 10:
        lr = 0.001
    else:
        lr = 0.0001
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    start_epoch = 1

    for epoch in range(start_epoch, args.epoch + start_epoch):
        print('This is %d-th epoch' % epoch)
        total_train_loss = 0
        total_test_loss = 0
        adjust_learning_rate(optimizer, epoch)

        for batch_id, (imgL, imgR, disp_L) in enumerate(tqdm(trainLoader)):
            train_loss = train(imgL, imgR, disp_L)
            total_train_loss += train_loss
        avg_train_loss = total_train_loss / len(trainLoader)
        print('Epoch %d average training loss = %.3f' % (epoch, avg_train_loss))

        for batch_id, (imgL, imgR, disp_L) in enumerate(tqdm(testLoader)):
            test_loss = test(imgL, imgR, disp_L)
            total_test_loss += test_loss
        avg_test_loss = total_test_loss / len(testLoader)
        print('Epoch %d total test loss = %.3f' % (epoch, avg_test_loss))

        state = {'net': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}

        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_model_path = args.save_path + 'checkpoint_{}.tar'.format(epoch)
        torch.save(state, save_model_path)

        torch.cuda.empty_cache()

    print('Training Finished!')


if __name__ == '__main__':
    main()

