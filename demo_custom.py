# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import numpy as np
import os
from scipy.special import softmax
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts_custom import parser
from ops import dataset_config_custom
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config_custom.return_dataset(args.dataset,
                                                                                                      args.modality)
    print("num_class: ", num_class)
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    args.resume = "/workdir/temporal-shift-module/output/checkpoint/TSM_custom_dataset_RGB_resnet50_avg_segment8_e50/2021-07-24-09.41.20/ckpt.best.pth.tar"
    print(("=> loading checkpoint '{}'".format(args.resume)))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    best_prec1 = checkpoint['best_prec1']
    print("best_prec1: ", best_prec1)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    args.evaluate = True

    # input
    video_path = '/workdir/datasets/video_classify/images/0/00005-00_06_40-00_06_48'
    # label 0
    indices = [ 18,  27,  60,  85, 115, 132, 172, 200]

    t = time.time()
    # load images
    image_tmpl = 'img_{:05d}.jpg'
    images = list()
    for seg_ind in indices:
        p = int(seg_ind)
        seg_imgs = [Image.open(os.path.join(video_path, image_tmpl.format(p))).convert('RGB')]
        images.extend(seg_imgs)
    # torch format tensor
    transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ])

    process_data = transform(images).unsqueeze(dim=0)
    output = model(process_data.cuda())
    output = output.cpu().detach().numpy()
    output = softmax(output, axis=1)
    label = np.argmax(output, axis=1)
    print("output: ", output)
    print("label: ", label)
    t = time.time() - t
    print("time: {:.2f}".format(t * 1000))


if __name__ == '__main__':
    main()
