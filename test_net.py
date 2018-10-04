# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
import pydicom
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from datetime import datetime
import pdb
import sys

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--arch', dest='arch', default='rcnn', choices=['rcnn', 'rfcn', 'couplenet'])
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="save",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--as', dest='anchor_scales',
                        help='anchor_scales',
                        default=3, type=int)
    parser.add_argument('--conf', dest='min_conf',
                        help='Min confidence score of bbox',
                        default=0.5, type=float)
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    # Import network definition
    if args.arch == 'rcnn':
        from model.faster_rcnn.vgg16 import vgg16
        from model.faster_rcnn.resnet import resnet
    elif args.arch == 'rfcn':
        from model.rfcn.resnet_atrous import resnet
    elif args.arch == 'couplenet':
        from model.couplenet.resnet_atrous import resnet

    # Import ROI functions as per dataset
    if args.dataset == 'kaggle_pna':
        from roi_data_layer.pnaRoiBatchLoader import roibatchLoader
        from roi_data_layer.pna_roidb import combined_roidb

        image_read_func = lambda path: pydicom.dcmread(path).pixel_array
    else:
        from roi_data_layer.roibatchLoader import roibatchLoader
        from roi_data_layer.roidb import combined_roidb

        image_read_func = lambda path: cv2.imread(path)

    print('Called with args:')
    print(args)

    # Warning to use cuda if available
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Anchor settings: ANCHOR_SCALES: [8, 16, 32] or [4, 8, 16, 32]
    if args.anchor_scales == 3:
        scales = [8, 16, 32]
    elif args.anchor_scales == 4:
        scales = [4, 8, 16, 32]
    else:
        scales = [8, 16, 32]

    # Dataset related settings: MAX_NUM_GT_BOXES: 20, 30, 50
    np.random.seed(cfg.RNG_SEED)

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', str(scales), 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "kaggle_pna":
        args.imdb_name = "pna_2018_trainval"
        args.imdbval_name = "pna_2018_test"
        args.set_cfgs = ['ANCHOR_SCALES', str(scales), 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False

    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)
    imdb.sub_mode = True
    print('{:d} roidb entries'.format(len(roidb)))

    # Trained network weights path
    input_dir = args.load_dir + "/" + args.arch + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             '{}_{}_{}_{}.pth'.format(args.arch, args.checksession, args.checkepoch, args.checkpoint))

    # Initialize the network:
    if args.net == 'vgg16':
        # model = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
        print("Pretrained model is not downloaded and network is not used")
    elif args.net == 'res18':
        model = resnet(imdb.classes, 18, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res34':
        model = resnet(imdb.classes, 34, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        model = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        model = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        model = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    # Create network architecture
    model.create_architecture()

    # Load pre-trained network weights
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    model.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # Initialize the tensor holder
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # Copy tensors in CUDA memory
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # Make variable
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)

    # Set cuda usage
    if args.cuda:
        cfg.CUDA = True

    # Copy network to CUDA memroy
    if args.cuda:
        model.cuda()

    # Start test or evaluation
    start = time.time()
    max_per_image = 100

    # Visualize output bounding boxes
    vis = args.vis
    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = args.arch + '_' + args.net
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0,  # args.num_workers
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    # Turn on model evaluation mode, i.e. train=False
    model.eval()

    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    dataset.resize_batch()
    for i in range(num_images):

        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = model(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:

            # Apply bounding-box regression deltas

            box_deltas = bbox_pred.data

            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(args.batch_size, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(args.batch_size, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        if vis:
            im = image_read_func(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:  # tensor.numel() -> returns number of elements in tensor
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:  # Find any object
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            cv2.imwrite('result.png', im2show)
            pdb.set_trace()
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    print('Kaggle submission file')

    if args.dataset == 'kaggle_pna':
        cipher = {'rcnn': 'alpha', 'rfcn': 'beta', 'couplenet': 'gamma'}
        created = datetime.now().strftime("%Y%m%d%H%M")
        sub_file = cipher[args.arch] + '_' + created + '_cls-{}_submission.txt'
        print('Submit file that ends with "_cls-3_submission.txt" file.')
        submission_file = os.path.join(output_dir, sub_file)
        imdb.write_kaggle_submission_file(all_boxes, submission_file, min_conf=args.min_conf)

    # with open(os.path.join(output_dir, 'all_boxes.pkl'), 'wb') as f:
    #     pickle.dump(all_boxes, f)
    # print('image_index: ', len(imdb.image_index))
    # print('all_boxes len: ', len(all_boxes))
    # print('all_boxes[0] len: ', len(all_boxes[0]))
    # print('all_boxes[1] len: ', len(all_boxes[1]))
    # print('all_boxes[2] len: ', len(all_boxes[2]))
    # print('all_boxes[3] len: ', len(all_boxes[3]))
    # print('all_boxes[1] shape: ', all_boxes[1].shape)
    # print('all_boxes[2] shape: ', all_boxes[2].shape)
    # print('all_boxes[3] shape: ', all_boxes[3].shape)
    # np.set_printoptions(precision=4, suppress=True)
    # for cls_idx, cls in enumerate(imdb.classes):
    #     if cls == '__background__':
    #         continue
    #     for im_idx, index in enumerate(imdb.image_index):
    #         dets = all_boxes[cls_idx][im_idx]
    #         print()
    #         probs = dets[:, -1]
    #         for p in probs:
    #             if p > 0.5:
    #                 if dets.shape[0] == 29:
    #                     print(dets)
    #         print()

    end = time.time()
    print("test time: %0.4fs" % (end - start))





# # --------------------------------------------------------
# # Tensorflow Faster R-CNN
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# # --------------------------------------------------------
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import _init_paths
# import os
# import sys
# import numpy as np
# import argparse
# import pprint
# import pdb
# import time
# import cv2
# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.optim as optim
# import pickle
# from roi_data_layer.roidb import combined_roidb
# from roi_data_layer.roibatchLoader import roibatchLoader
# from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
# from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
# from model.rpn.bbox_transform import bbox_transform_inv
# from model.utils.net_utils import save_net, load_net, vis_detections
#
# import pdb
#
# try:
#     xrange          # Python 2
# except NameError:
#     xrange = range  # Python 3
#
#
# def parse_args():
#     """
#     Parse input arguments
#     """
#     parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
#     parser.add_argument('--arch', dest='arch', default='rcnn', choices=['rcnn', 'rfcn', 'couplenet'])
#     parser.add_argument('--dataset', dest='dataset',
#                         help='training dataset',
#                         default='pascal_voc', type=str)
#     parser.add_argument('--cfg', dest='cfg_file',
#                         help='optional config file',
#                         default='cfgs/vgg16.yml', type=str)
#     parser.add_argument('--net', dest='net',
#                         help='vgg16, res50, res101, res152',
#                         default='res101', type=str)
#     parser.add_argument('--set', dest='set_cfgs',
#                         help='set config keys', default=None,
#                         nargs=argparse.REMAINDER)
#     parser.add_argument('--load_dir', dest='load_dir',
#                         help='directory to load models', default="save",
#                         type=str)
#     parser.add_argument('--cuda', dest='cuda',
#                         help='whether use CUDA',
#                         action='store_true')
#     parser.add_argument('--ls', dest='large_scale',
#                         help='whether use large imag scale',
#                         action='store_true')
#     parser.add_argument('--mGPUs', dest='mGPUs',
#                         help='whether use multiple GPUs',
#                         action='store_true')
#     parser.add_argument('--cag', dest='class_agnostic',
#                         help='whether perform class_agnostic bbox regression',
#                         action='store_true')
#     parser.add_argument('--parallel_type', dest='parallel_type',
#                         help='which part of model to parallel, 0: all, 1: model before roi pooling',
#                         default=0, type=int)
#     parser.add_argument('--checksession', dest='checksession',
#                         help='checksession to load model',
#                         default=1, type=int)
#     parser.add_argument('--checkepoch', dest='checkepoch',
#                         help='checkepoch to load network',
#                         default=1, type=int)
#     parser.add_argument('--checkpoint', dest='checkpoint',
#                         help='checkpoint to load network',
#                         default=10021, type=int)
#     parser.add_argument('--bs', dest='batch_size',
#                         help='batch_size',
#                         default=1, type=int)
#     parser.add_argument('--vis', dest='vis',
#                         help='visualization mode',
#                         action='store_true')
#     args = parser.parse_args()
#     return args
#
# lr = cfg.TRAIN.LEARNING_RATE
# momentum = cfg.TRAIN.MOMENTUM
# weight_decay = cfg.TRAIN.WEIGHT_DECAY
#
# if __name__ == '__main__':
#
#     args = parse_args()
#
#     if args.arch == 'rcnn':
#         from model.faster_rcnn.vgg16 import vgg16
#         from model.faster_rcnn.resnet import resnet
#     elif args.arch == 'rfcn':
#         from model.rfcn.resnet_atrous import resnet
#     elif args.arch == 'couplenet':
#         from model.couplenet.resnet_atrous import resnet
#
#     print('Called with args:')
#     print(args)
#
#     if torch.cuda.is_available() and not args.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#
#     np.random.seed(cfg.RNG_SEED)
#     if args.dataset == "pascal_voc":
#         args.imdb_name = "voc_2007_trainval"
#         args.imdbval_name = "voc_2007_test"
#         args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
#     elif args.dataset == "pascal_voc_0712":
#         args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
#         args.imdbval_name = "voc_2007_test"
#         args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
#     elif args.dataset == "coco":
#         args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
#         args.imdbval_name = "coco_2014_minival"
#         args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
#     elif args.dataset == "imagenet":
#         args.imdb_name = "imagenet_train"
#         args.imdbval_name = "imagenet_val"
#         args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
#     elif args.dataset == "vg":
#         args.imdb_name = "vg_150-50-50_minitrain"
#         args.imdbval_name = "vg_150-50-50_minival"
#         args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
#
#     args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
#
#     if args.cfg_file is not None:
#         cfg_from_file(args.cfg_file)
#     if args.set_cfgs is not None:
#         cfg_from_list(args.set_cfgs)
#
#     print('Using config:')
#     pprint.pprint(cfg)
#
#     cfg.TRAIN.USE_FLIPPED = False
#     imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
#     imdb.competition_mode(on=True)
#
#     print('{:d} roidb entries'.format(len(roidb)))
#
#     input_dir = args.load_dir + "/" + args.arch + "/" + args.net + "/" + args.dataset
#     if not os.path.exists(input_dir):
#         raise Exception('There is no input directory for loading network from ' + input_dir)
#     load_name = os.path.join(input_dir,
#                              'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
#
#     # initilize the network here.
#     if args.net == 'vgg16':
#         fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
#     elif args.net == 'res101':
#         fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
#     elif args.net == 'res50':
#         fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
#     elif args.net == 'res152':
#         fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
#     else:
#         print("network is not defined")
#         pdb.set_trace()
#
#     fasterRCNN.create_architecture()
#
#     print("load checkpoint %s" % (load_name))
#     checkpoint = torch.load(load_name)
#     fasterRCNN.load_state_dict(checkpoint['model'])
#     if 'pooling_mode' in checkpoint.keys():
#         cfg.POOLING_MODE = checkpoint['pooling_mode']
#
#
#     print('load model successfully!')
#     # initilize the tensor holder here.
#     im_data = torch.FloatTensor(1)
#     im_info = torch.FloatTensor(1)
#     num_boxes = torch.LongTensor(1)
#     gt_boxes = torch.FloatTensor(1)
#
#     # ship to cuda
#     if args.cuda:
#         im_data = im_data.cuda()
#         im_info = im_info.cuda()
#         num_boxes = num_boxes.cuda()
#         gt_boxes = gt_boxes.cuda()
#
#     # make variable
#     im_data = Variable(im_data, volatile=True)
#     im_info = Variable(im_info, volatile=True)
#     num_boxes = Variable(num_boxes, volatile=True)
#     gt_boxes = Variable(gt_boxes, volatile=True)
#
#     if args.cuda:
#         cfg.CUDA = True
#
#     if args.cuda:
#         fasterRCNN.cuda()
#
#     start = time.time()
#     max_per_image = 100
#
#     vis = args.vis
#
#     if vis:
#         thresh = 0.05
#     else:
#         thresh = 0.0
#
#     save_name = 'faster_rcnn_10'
#     num_images = len(imdb.image_index)
#     all_boxes = [[[] for _ in xrange(num_images)]
#                  for _ in xrange(imdb.num_classes)]
#
#     output_dir = get_output_dir(imdb, save_name)
#     dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
#                              imdb.num_classes, training=False, normalize = False)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
#                                              shuffle=False, num_workers=0,
#                                              pin_memory=True)
#
#     data_iter = iter(dataloader)
#
#     _t = {'im_detect': time.time(), 'misc': time.time()}
#     det_file = os.path.join(output_dir, 'detections.pkl')
#
#     fasterRCNN.eval()
#     empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
#     dataset.resize_batch()
#     for i in range(num_images):
#
#         data = next(data_iter)
#         im_data.data.resize_(data[0].size()).copy_(data[0])
#         im_info.data.resize_(data[1].size()).copy_(data[1])
#         gt_boxes.data.resize_(data[2].size()).copy_(data[2])
#         num_boxes.data.resize_(data[3].size()).copy_(data[3])
#
#         det_tic = time.time()
#         rois, cls_prob, bbox_pred, \
#         rpn_loss_cls, rpn_loss_box, \
#         RCNN_loss_cls, RCNN_loss_bbox, \
#         rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
#
#         scores = cls_prob.data
#         boxes = rois.data[:, :, 1:5]
#
#         if cfg.TEST.BBOX_REG:
#             # Apply bounding-box regression deltas
#             box_deltas = bbox_pred.data
#             if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
#                 # Optionally normalize targets by a precomputed mean and stdev
#                 if args.class_agnostic:
#                     box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
#                                  + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
#                     box_deltas = box_deltas.view(args.batch_size, -1, 4)
#                 else:
#                     box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
#                                  + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
#                     box_deltas = box_deltas.view(args.batch_size, -1, 4 * len(imdb.classes))
#
#             pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
#             pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
#         else:
#             # Simply repeat the boxes, once for each class
#             pred_boxes = np.tile(boxes, (1, scores.shape[1]))
#
#         pred_boxes /= data[1][0][2]
#
#         scores = scores.squeeze()
#         pred_boxes = pred_boxes.squeeze()
#         det_toc = time.time()
#         detect_time = det_toc - det_tic
#         misc_tic = time.time()
#         if vis:
#             im = cv2.imread(imdb.image_path_at(i))
#             im2show = np.copy(im)
#         for j in xrange(1, imdb.num_classes):
#             inds = torch.nonzero(scores[:,j]>thresh).view(-1)
#             # if there is det
#             if inds.numel() > 0:
#                 cls_scores = scores[:,j][inds]
#                 _, order = torch.sort(cls_scores, 0, True)
#                 if args.class_agnostic:
#                     cls_boxes = pred_boxes[inds, :]
#                 else:
#                     cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
#
#                 cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
#                 # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
#                 cls_dets = cls_dets[order]
#                 keep = nms(cls_dets, cfg.TEST.NMS)
#                 cls_dets = cls_dets[keep.view(-1).long()]
#                 if vis:
#                     im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
#                 all_boxes[j][i] = cls_dets.cpu().numpy()
#             else:
#                 all_boxes[j][i] = empty_array
#
#         # Limit to max_per_image detections *over all classes*
#         if max_per_image > 0:
#             image_scores = np.hstack([all_boxes[j][i][:, -1]
#                                       for j in xrange(1, imdb.num_classes)])
#             if len(image_scores) > max_per_image:
#                 image_thresh = np.sort(image_scores)[-max_per_image]
#                 for j in xrange(1, imdb.num_classes):
#                     keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
#                     all_boxes[j][i] = all_boxes[j][i][keep, :]
#
#         misc_toc = time.time()
#         nms_time = misc_toc - misc_tic
#
#         sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
#                          .format(i + 1, num_images, detect_time, nms_time))
#         sys.stdout.flush()
#
#         if vis:
#             cv2.imwrite('result.png', im2show)
#             pdb.set_trace()
#             #cv2.imshow('test', im2show)
#             #cv2.waitKey(0)
#
#     with open(det_file, 'wb') as f:
#         pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
#
#     print('Evaluating detections')
#     imdb.evaluate_detections(all_boxes, output_dir)
#
#     end = time.time()
#     print("test time: %0.4fs" % (end - start))
