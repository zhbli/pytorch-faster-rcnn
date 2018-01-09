# lzb: visualize the false detection
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
try:
    import cPickle as pickle
except ImportError:
    import pickle
import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import time
from nets.vgg16 import vgg16

if __name__ == '__main__':
    filename = 'output/bird_false_positive.txt'
    filename_rois = 'output/bird_false_positive_rois.txt'
    f = open(filename, 'rb')
    d = pickle.load(f)
    f.close()
    f = open(filename_rois, 'rb')
    rois = pickle.load(f)
    f.close()

    for i in range(len(d)):
        if len(d[i]) == 0:
            break
        im_file = '/data/zhbli/VOCdevkit/VOC2007/JPEGImages/%s.jpg'%(d[i][0][0])
        im = cv2.imread(im_file)
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        bbox = d[i][0][3:]
        roi = rois[i][0][3:]
        score = d[i][2]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.add_patch(
            plt.Rectangle((roi[0], roi[1]),
                          roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='yellow', linewidth=3.5)
        )
        for j in range(len(d[i][1])):
            gtbox = d[i][1][j]
            ax.add_patch(
                plt.Rectangle((gtbox[0], gtbox[1]),
                              gtbox[2] - gtbox[0],
                              gtbox[3] - gtbox[1], fill=False,
                              edgecolor='green', linewidth=3.5)
            )

        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(d[i][0][0]) + ' score={:f}'.format(score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=10, color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.show()