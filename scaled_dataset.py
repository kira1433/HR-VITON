#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import json
import argparse
import os.path as osp
import numpy as np

class ScaledDataset(data.Dataset):
    """
        Dataset for Scaled-Viton.
    """
    def __init__(self, opt):
        super(ScaledDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = dict()
        self.c_names['paired'] = im_names
        self.c_names['unpaired'] = c_names

    def name(self):
        return "ScaledDataset"
    
    @staticmethod
    def get_params(img, scale):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        _, height, width = F.get_dimensions(img)
        rand_scale = scale[0] + torch.rand(1) * (scale[1] - scale[0])
        h = int(height * rand_scale)
        w = int(width * rand_scale)
        i = torch.randint(0, height - h + 1, size=(1, )).item()
        j = torch.randint(0, width - w + 1, size=(1, )).item()
        return i, j, h, w

    def ScaledResizedCrop(self, img, i, j, h, w):
        img = F.resized_crop(img, i, j, h, w, (self.fine_height, self.fine_width), Image.BILINEAR)
        return img

    def __getitem__(self, index):
        im_name = self.im_names[index]
        im_name = 'image/' + im_name
        c_name = {}
        c = {}
        cm = {}
        for key in self.c_names:
            c_name[key] = self.c_names[key][index]
            c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
            c[key] = transforms.Resize(self.fine_width, interpolation=2)(c[key])
            cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
            cm[key] = transforms.Resize(self.fine_width, interpolation=0)(cm[key])
            
            c[key] = self.transform(c[key])  # [-1,1]
            cm_array = np.array(cm[key])
            cm_array = (cm_array >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_array)  # [0,1]
            cm[key].unsqueeze_(0)
        
        # person image
        im_pil_big = Image.open(osp.join(self.data_path, im_name))
        # ix,jx,hx,wx = 0,0,self.fine_height,self.fine_width
        ix,jx,hx,wx = self.get_params(im_pil_big, scale=[0.7, 1.0])
        im_pil_big = self.ScaledResizedCrop(im_pil_big, ix, jx, hx, wx)
        # im_pil_big.show()
        im_pil = transforms.Resize(self.fine_width, interpolation=2)(im_pil_big)
        im = self.transform(im_pil)

        # load parsing image
        parse_name = im_name.replace('image', 'image-parse-v3').replace('.jpg', '.png')
        im_parse_pil_big = Image.open(osp.join(self.data_path, parse_name))
        im_parse_pil_big = self.ScaledResizedCrop(im_parse_pil_big, ix, jx, hx, wx)
        # im_parse_pil_big.show()
        im_parse_pil = transforms.Resize(self.fine_width, interpolation=0)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        im_parse = self.transform(im_parse_pil.convert('RGB'))

        # parse map
        labels = {
            0:  ['background',  [0, 10]],
            1:  ['hair',        [1, 2]],
            2:  ['face',        [4, 13]],
            3:  ['upper',       [5, 6, 7]],
            4:  ['bottom',      [9, 12]],
            5:  ['left_arm',    [14]],
            6:  ['right_arm',   [15]],
            7:  ['left_leg',    [16]],
            8:  ['right_leg',   [17]],
            9:  ['left_shoe',   [18]],
            10: ['right_shoe',  [19]],
            11: ['socks',       [8]],
            12: ['noise',       [3, 11]]
        }
        
        parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]
                
        parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i
                
        # load image-parse-agnostic
        image_parse_agnostic = Image.open(osp.join(self.data_path, parse_name.replace('image-parse-v3', 'image-parse-agnostic-v3.2')))
        image_parse_agnostic = self.ScaledResizedCrop(image_parse_agnostic, ix, jx, hx, wx)
        # image_parse_agnostic.show()
        image_parse_agnostic = transforms.Resize(self.fine_width, interpolation=0)(image_parse_agnostic)
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        image_parse_agnostic = self.transform(image_parse_agnostic.convert('RGB'))

        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]        
                
                     
        # parse cloth & parse cloth mask
        pcm = new_parse_map[3:4]
        im_c = im * pcm + (1 - pcm)

        # load pose points
        pose_name = im_name.replace('image', 'openpose_img').replace('.jpg', '_rendered.png')
        pose_map = Image.open(osp.join(self.data_path, pose_name))
        pose_map = self.ScaledResizedCrop(pose_map, ix, jx, hx, wx)
        # pose_map.show()
        pose_map = transforms.Resize(self.fine_width, interpolation=2)(pose_map)
        pose_map = self.transform(pose_map)  # [-1,1]
        
        # load densepose
        densepose_name = im_name.replace('image', 'image-densepose')
        densepose_map = Image.open(osp.join(self.data_path, densepose_name))
        densepose_map = self.ScaledResizedCrop(densepose_map, ix, jx, hx, wx)
        # densepose_map.show()
        densepose_map = transforms.Resize(self.fine_width, interpolation=2)(densepose_map)
        densepose_map = self.transform(densepose_map)  # [-1,1]

        # agnostic
        agnostic  = Image.open(osp.join(self.data_path, im_name.replace('image', 'agnostic-v3.2')))
        agnostic = self.ScaledResizedCrop(agnostic, ix, jx, hx, wx)
        # agnostic.show()
        agnostic = transforms.Resize(self.fine_width, interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            # intput 1 (clothfloww)
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            # intput 2 (segnet)
            'parse_agnostic': new_parse_agnostic_map,
            'densepose': densepose_map,
            'pose': pose_map,       # for conditioning
            # generator input
            'agnostic' : agnostic,
            # GT
            'parse_onehot' : parse_onehot,  # Cross Entropy
            'parse': new_parse_map, # GAN Loss real
            'pcm': pcm,             # L1 Loss & vis
            'parse_cloth': im_c,    # VGG Loss & vis
            # visualization & GT
            'image':    im,         # for visualization
            }

        return result

    def __len__(self):
        return len(self.im_names)
    

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', help='use amp')
    # Cuda availability
    parser.add_argument('--cuda',default=False, help='cuda or cpu')

    parser.add_argument('--test_name', type=str, default='test', help='test name')
    parser.add_argument("--dataroot", default="./data/zalando-hd-resize")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--output_dir", type=str, default="./Output")
    parser.add_argument("--datasetting", default="unpaired")
    parser.add_argument("--fine_width", type=int, default=768)
    parser.add_argument("--fine_height", type=int, default=1024)

    parser.add_argument('--tensorboard_dir', type=str, default='./data/zalando-hd-resize/tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, default='./eval_models/weights/v0.1/mtviton.pth', help='tocg checkpoint')
    parser.add_argument('--gen_checkpoint', type=str, default='./eval_models/weights/v0.1/gen.pth', help='G checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    parser.add_argument('--gen_semantic_nc', type=int, default=7, help='# of input label classes without unknown class')
    
    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    
    # training
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
        
    # Hyper-parameters
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--occlusion', action='store_true', help="Occlusion handling")

    # generator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance', help='instance normalization or batch normalization')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='most', # normal: 256, more: 512
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

    opt = parser.parse_args()
    return opt

def main():
    try:
        opt = get_opt()
        print(opt)

        dataset = ScaledDataset(opt)
        for i in range(1):
            item = dataset.__getitem__(i)
            print(item['c_name'])
            print(item['im_name'])
            # print(item['cloth'])
            # print(item['cloth_mask'])
            # print(item['parse_agnostic'])
            # print(item['densepose'])
            # print(item['pose'])
            # print(item['agnostic'])
            # print(item['parse_onehot'])
            # print(item['parse'])
            # print(item['pcm'])
            # print(item['parse_cloth'])
            # print(item['image'])
            print('------------------')

    except Exception as error:
        print(error)

if __name__ == '__main__':
    main()