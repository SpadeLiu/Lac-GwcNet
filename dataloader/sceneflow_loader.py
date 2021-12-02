import os
from PIL import Image
from dataloader import readpfm as rp
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random

IMG_EXTENSIONS= [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def sf_loader(filepath):

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_cleanpass') > -1]
    disparity = [disp for disp in classes if disp.find('disparity') > -1]

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    monkaa_img = filepath + [x for x in image if 'monkaa' in x][0]
    monkaa_disp = filepath + [x for x in disparity if 'monkaa' in x][0]
    monkaa_dir = os.listdir(monkaa_img)
    for dd in monkaa_dir:
        left_path = monkaa_img + '/' + dd + '/left/'
        right_path = monkaa_img + '/' + dd + '/right/'
        disp_path = monkaa_disp + '/' + dd + '/left/'

        left_imgs = os.listdir(left_path)
        for img in left_imgs:
            img_path = os.path.join(left_path, img)
            if is_image_file(img_path):
                all_left_img.append(img_path)
                all_right_img.append(os.path.join(right_path, img))
                all_left_disp.append(disp_path + img.split(".")[0] + '.pfm')

    flying_img = filepath + [x for x in image if 'flying' in x][0]
    flying_disp = filepath + [x for x in disparity if 'flying' in x][0]
    fimg_train = flying_img + '/TRAIN/'
    fimg_test = flying_img + '/TEST/'
    fdisp_train = flying_disp + '/TRAIN/'
    fdisp_test = flying_disp + '/TEST/'
    fsubdir = ['A', 'B', 'C']

    for dd in fsubdir:
        imgs_path = fimg_train + dd + '/'
        disps_path = fdisp_train + dd + '/'
        imgs = os.listdir(imgs_path)
        for cc in imgs:
            left_path = imgs_path + cc + '/left/'
            right_path = imgs_path + cc + '/right/'
            disp_path = disps_path + cc + '/left/'

            left_imgs = os.listdir(left_path)
            for img in left_imgs:
                img_path = os.path.join(left_path, img)
                if is_image_file(img_path):
                    all_left_img.append(img_path)
                    all_right_img.append(os.path.join(right_path, img))
                    all_left_disp.append(disp_path + img.split(".")[0] + '.pfm')

    for dd in fsubdir:
        imgs_path = fimg_test + dd + '/'
        disps_path = fdisp_test + dd + '/'
        imgs = os.listdir(imgs_path)
        for cc in imgs:
            left_path = imgs_path + cc + '/left/'
            right_path = imgs_path + cc + '/right/'
            disp_path = disps_path + cc + '/left/'

            left_imgs = os.listdir(left_path)
            for img in left_imgs:
                img_path = os.path.join(left_path, img)
                if is_image_file(img_path):
                    test_left_img.append(img_path)
                    test_right_img.append(os.path.join(right_path, img))
                    test_left_disp.append(disp_path + img.split(".")[0] + '.pfm')

    driving_img = filepath + [x for x in image if 'driving' in x][0]
    driving_disp = filepath + [x for x in disparity if 'driving' in x][0]
    dsubdir1 = ['15mm_focallength', '35mm_focallength']
    dsubdir2 = ['scene_backwards', 'scene_forwards']
    dsubdir3 = ['fast', 'slow']
    for d in dsubdir1:
        img_path1 = driving_img + '/' + d + '/'
        disp_path1 = driving_disp + '/' + d + '/'
        for dd in dsubdir2:
            img_path2 = img_path1 + dd + '/'
            disp_path2 = disp_path1 + dd + '/'
            for ddd in dsubdir3:
                img_path3 = img_path2 + ddd + '/'
                disp_path3 = disp_path2 + ddd + '/'

                left_path = img_path3 + 'left/'
                right_path = img_path3 + 'right/'
                disp_path = disp_path3 + 'left/'

                left_imgs = os.listdir(left_path)
                for img in left_imgs:
                    img_path = os.path.join(left_path, img)
                    if is_image_file(img_path):
                        all_left_img.append(img_path)
                        all_right_img.append(os.path.join(right_path, img))
                        all_left_disp.append(disp_path + img.split(".")[0] + '.pfm')

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


def img_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class myDataset(data.Dataset):

    def __init__(self, left, right, left_disp, training, imgloader=img_loader, dploader = disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disp
        self.imgloader = imgloader
        self.dploader = dploader
        self.training = training
        self.img_transorm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.imgloader(left)
        right_img = self.imgloader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        if self.training:
            w, h = left_img.size
            tw, th = 512, 256
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1+tw, y1+th))
            right_img = right_img.crop((x1, y1, x1+tw, y1+th))
            dataL = dataL[y1:y1+th, x1:x1+tw]

            left_img = self.img_transorm(left_img)
            right_img = self.img_transorm(right_img)

            return left_img, right_img, dataL

        else:
            w, h = left_img.size
            left_img = left_img.crop((w-960, h-544, w, h))
            right_img = right_img.crop((w-960, h-544, w, h))

            left_img = self.img_transorm(left_img)
            right_img = self.img_transorm(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)

