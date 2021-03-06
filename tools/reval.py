#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# Reval = re-eval. Re-evaluate saved detections.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import apply_nms
from model.config import cfg
from datasets.factory import get_imdb
import pickle
import os, sys, argparse
import numpy as np

# v3.2
import global_var
# v3.2


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Re-evaluate results')
  parser.add_argument('output_dir', nargs=1, help='results directory',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to re-evaluate',
                      default='voc_2007_test', type=str)
  parser.add_argument('--matlab', dest='matlab_eval',
                      help='use matlab for evaluation',
                      action='store_true')
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                      action='store_true')
  parser.add_argument('--nms', dest='apply_nms', help='apply nms',
                      action='store_true')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def from_dets(imdb_name, output_dir, args):
  imdb = get_imdb(imdb_name)
  imdb.competition_mode(args.comp_mode)
  imdb.config['matlab_eval'] = args.matlab_eval
  with open(os.path.join(output_dir, 'detections.pkl'), 'rb') as f:
    dets = pickle.load(f)
  with open(os.path.join(output_dir, 'detections_rois.pkl'), 'rb') as f:
    rois = pickle.load(f)

  if args.apply_nms:
    assert False, 'No nms step in test.py -> test_net'
    print('Applying NMS to all detections')
    nms_dets = apply_nms(dets, cfg.TEST.NMS)
  else:
    nms_dets = dets

  print('Evaluating detections')
  imdb.evaluate_detections(nms_dets, output_dir, rois)


if __name__ == '__main__':

  # v3.2
  global_var.global_reval_version = 3.2
  # v3.2

  args = parse_args()

  output_dir = os.path.abspath(args.output_dir[0])
  imdb_name = args.imdb_name
  from_dets(imdb_name, output_dir, args)
