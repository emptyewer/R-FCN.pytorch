from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch.nn as nn

from model.couplenet.couplenet import CoupleNet
from model.utils.config import cfg

# Pre-trained weights: https://github.com/jrzech/reproduce-chexnet

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
from collections import OrderedDict
import torchvision

__all__ = ['DenseNet121', 'densenet121']

def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet121(classCount=14, pretrained=pretrained, **kwargs).cuda()
    if pretrained:
        import datasets
        model_repo_path = os.path.dirname(os.path.dirname(os.path.dirname(datasets.__file__)))
        # CHEXNET WEIGHTS
        pretrained_model_path = os.path.join(model_repo_path, 'data/pretrained_model/m-25012018-123527.pth.tar')
        model = torch.nn.DataParallel(model).cuda()
        modelCheckpoint = torch.load(pretrained_model_path)
        model.load_state_dict(modelCheckpoint['state_dict'])
    return model

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())


    def forward(self, x):
        x = self.densenet121(x)
        return x

class chexnet(CoupleNet):
    def __init__(self, classes, num_layers=121, pretrained=False, class_agnostic=False):
        self.num_layers = num_layers
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        CoupleNet.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        densenet = eval('densenet{}({})'.format(self.num_layers, self.pretrained))

        # Build densene
        self.RCNN_base = nn.Sequential(
            densenet.densenet121.features.conv0,
            densenet.densenet121.features.norm0,
            # densenet.features.relu0,
            # densenet.features.pool0,
            densenet.densenet121.features.denseblock1,
            densenet.densenet121.features.transition1,
            densenet.densenet121.features.denseblock2,
            densenet.densenet121.features.transition2,
            densenet.densenet121.features.denseblock3,
            densenet.densenet121.features.transition3,
            densenet.densenet121.features.denseblock4,
            densenet.densenet121.features.norm5,
        )

        self.RCNN_conv_1x1 = nn.Conv2d(in_channels=1024, out_channels=1024,
                  kernel_size=1, stride=1, padding=0, bias=False)

        self.RCNN_conv_new = nn.Sequential(
            self.RCNN_conv_1x1,
            nn.ReLU()
        )

        # Local feature layers
        if self.class_agnostic:
            self.RCNN_local_bbox_base = nn.Conv2d(in_channels=1024,
                                                  out_channels=4 * cfg.POOLING_SIZE * cfg.POOLING_SIZE,
                                                  kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.RCNN_local_bbox_base = nn.Conv2d(in_channels=1024,
                                                  out_channels=4 * self.n_classes * cfg.POOLING_SIZE * cfg.POOLING_SIZE,
                                                  kernel_size=1, stride=1, padding=0, bias=False)
        self.RCNN_local_cls_base = nn.Conv2d(in_channels=1024,
                                             out_channels=self.n_classes * cfg.POOLING_SIZE * cfg.POOLING_SIZE,
                                             kernel_size=1, stride=1, padding=0, bias=False)
        self.RCNN_local_cls_fc = nn.Conv2d(in_channels=self.n_classes, out_channels=self.n_classes,
                                           kernel_size=1, stride=1, padding=0, bias=False)

        # Global feature layers
        self.RCNN_global_base = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=7, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.RCNN_global_cls = nn.Conv2d(in_channels=1024, out_channels=self.n_classes, kernel_size=1, stride=1,
                                         padding=0, bias=False)

        if self.class_agnostic:
            self.RCNN_global_bbox = nn.Conv2d(in_channels=1024, out_channels=4, kernel_size=1, stride=1, padding=0,
                                              bias=False)
        else:
            self.RCNN_global_bbox = nn.Conv2d(in_channels=1024, out_channels=4 * self.n_classes, kernel_size=1,
                                              stride=1, padding=0,
                                              bias=False)

        # Fix blocks
        for p in self.RCNN_base[0].parameters(): p.requires_grad = False
        for p in self.RCNN_base[1].parameters(): p.requires_grad = False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_base[6].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_base[5].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_base[4].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_conv_new.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            for fix_layer in range(6, 3 + cfg.RESNET.FIXED_BLOCKS, -1):
                self.RCNN_base[fix_layer].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.RCNN_conv_new.apply(set_bn_eval)

    if __name__ == '__main__':
        import torch
        import numpy as np
        from torch.autograd import Variable

        input = torch.randn(3, 3, 600, 800)

        model = resnet101().cuda()
        input = Variable(input.cuda())
        out = model(input)
        print(out.size())