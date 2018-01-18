# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False,
             use_diff=False, roipath=''):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')

  # v3.0
  det_results = {}
  # v3.0

  # extract gt objects for this class
  class_recs = {}
  npos = 0
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    truncated = np.array([x['truncated'] for x in R]).astype(np.bool)
    if use_diff:
      difficult = np.array([False for x in R]).astype(np.bool)
    else:
      difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}

    # v3.0
    det_results[imagename] = {}
    det_results[imagename]['gt'] = {}
    det_results[imagename]['det'] = {}
    det_results[imagename]['det']['bbox'] = np.zeros([0,4])
    det_results[imagename]['det']['score'] = []
    det_results[imagename]['det']['result_info'] = []
    det_results[imagename]['det']['overlap'] = []
    det_results[imagename]['gt']['bbox'] = bbox
    det_results[imagename]['gt']['difficult'] = difficult
    det_results[imagename]['gt']['truncated'] = truncated
    det_results[imagename]['gt']['detected'] = det
    # v3.0

  # read dets
  detfile = detpath.format(classname)
  roifile = roipath.format(classname)
  with open(detfile, 'r') as f:
    lines = f.readlines()
  with open(roifile, 'r') as f:
    lines_rois = f.readlines()

  splitlines = [x.strip().split(' ') for x in lines]
  splitlines_rois = [x.strip().split(' ') for x in lines_rois]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
  rois = np.array([[float(z) for z in x[2:]] for x in splitlines_rois])

  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  #zhbli(1): write false negative info
  false_positive_file_name = 'output/%s_false_positive.txt' % (classname)
  false_positive_rois_file_name = 'output/%s_false_positive_rois.txt' % (classname)
  false_positive_file = open(false_positive_file_name, 'wb')
  false_positive_rois_file = open(false_positive_rois_file_name, 'wb')
  fp_table = [[] for i in range(npos)]
  fp_table_rois = [[] for i in range(npos)]
  fp_idx = 0

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    rois = rois[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      roi = rois[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1

            # v3.0
            det_results[image_ids[d]]['gt']['detected'][jmax] = 1  # the jmax'th gt is detected.
            det_results[image_ids[d]]['det']['bbox'] = np.append(det_results[image_ids[d]]['det']['bbox'], np.expand_dims(bb, 0), axis=0)
            det_results[image_ids[d]]['det']['score'].append(-sorted_scores[d])
            det_results[image_ids[d]]['det']['result_info'].append('correct')
            det_results[image_ids[d]]['det']['overlap'].append(ovmax)
            # v3.0

          else:
            fp[d] = 1.

            # v3.0
            if -sorted_scores[d] > 0.1:
                det_results[image_ids[d]]['det']['bbox'] = np.append(det_results[image_ids[d]]['det']['bbox'],
                                                                     np.expand_dims(bb, 0), axis=0)
                det_results[image_ids[d]]['det']['score'].append(-sorted_scores[d])
                det_results[image_ids[d]]['det']['result_info'].append('err_repeat')
                det_results[image_ids[d]]['det']['overlap'].append(ovmax)
            # v3.0

            #zhbli(2): write false negative info
            if d < npos:
                fp_table[fp_idx] = [[image_ids[d], classname, 'err_repeat'] + bb.tolist(), BBGT.tolist(), -sorted_scores[d].tolist()]
                fp_table_rois[fp_idx] = [[image_ids[d], classname, 'err_repeat'] + roi.tolist(), BBGT.tolist(), -sorted_scores[d].tolist()]
                fp_idx = fp_idx + 1
      else:
        fp[d] = 1.

        # v3.0
        if -sorted_scores[d] > 0.1:
            det_results[image_ids[d]]['det']['bbox'] = np.append(det_results[image_ids[d]]['det']['bbox'],
                                                                 np.expand_dims(bb, 0), axis=0)
            det_results[image_ids[d]]['det']['score'].append(-sorted_scores[d])
            det_results[image_ids[d]]['det']['result_info'].append('err_lowIoU')
            det_results[image_ids[d]]['det']['overlap'].append(ovmax)
        # v3.0

        # zhbli(3): write false negative info
        if d < npos:
            fp_table[fp_idx] = [[image_ids[d], classname, 'err_lowIoU'] + bb.tolist(), BBGT.tolist(), -sorted_scores[d].tolist()]
            fp_table_rois[fp_idx] = [[image_ids[d], classname, 'err_repeat'] + roi.tolist(), BBGT.tolist(), -sorted_scores[d].tolist()]
            fp_idx = fp_idx + 1

    # zhbli(4): write false negative info
    pickle.dump(fp_table, false_positive_file)
    pickle.dump(fp_table_rois, false_positive_rois_file)
    false_positive_file.close()
    false_positive_rois_file.close()

  # v3.1
  if not os.path.exists('/data/zhbli/VOCdevkit/results/VOC2007/vgg16_faster-rcnn/{:s}'.format(classname)):
      os.mkdir('/data/zhbli/VOCdevkit/results/VOC2007/vgg16_faster-rcnn/{:s}'.format(classname))

  # no missed, no fp
  if not os.path.exists('/data/zhbli/VOCdevkit/results/VOC2007/vgg16_faster-rcnn/{:s}/correct'.format(classname)):
      os.mkdir('/data/zhbli/VOCdevkit/results/VOC2007/vgg16_faster-rcnn/{:s}/correct'.format(classname))

  # no gt, fp
  if not os.path.exists('/data/zhbli/VOCdevkit/results/VOC2007/vgg16_faster-rcnn/{:s}/no_gt_fp'.format(classname)):
      os.mkdir('/data/zhbli/VOCdevkit/results/VOC2007/vgg16_faster-rcnn/{:s}/no_gt_fp'.format(classname))

  # other
  if not os.path.exists('/data/zhbli/VOCdevkit/results/VOC2007/vgg16_faster-rcnn/{:s}/other'.format(classname)):
      os.mkdir('/data/zhbli/VOCdevkit/results/VOC2007/vgg16_faster-rcnn/{:s}/other'.format(classname))
  # v3.1

  # v3.0
  for i in range(len(det_results)):
  # for every img
      img_name = str(i).zfill(6)
      if not det_results.__contains__(img_name):
          continue
      gt_num = det_results[img_name]['gt']['bbox'].shape[0]
      det_num = det_results[img_name]['det']['bbox'].shape[0]
      if gt_num != 0 or det_num != 0:
          print('saving img: {:s}'.format(img_name))
          im_file = '/data/zhbli/VOCdevkit/VOC2007/JPEGImages/%s.jpg' % img_name
          im = cv2.imread(im_file)
          im = im[:, :, (2, 1, 0)]
          fig, ax = plt.subplots(figsize=(12, 12))
          ax.imshow(im, aspect='equal')
      else:
          continue  # If the img has no gt and no det, will not save this img.

      if gt_num != 0:
          for j in range(gt_num):
              bbox = det_results[img_name]['gt']['bbox'][j]
              difficult = det_results[img_name]['gt']['difficult'][j]
              truncated = det_results[img_name]['gt']['truncated'][j]
              detected = det_results[img_name]['gt']['detected'][j]
              if detected == 1:  # If the gt is found
                  ax.add_patch(
                      plt.Rectangle((bbox[0], bbox[1]),
                                    bbox[2] - bbox[0],
                                    bbox[3] - bbox[1], fill=False, edgecolor='blue',
                                    linewidth=1.5, linestyle='solid')
                  )
                  ax.text(bbox[0], bbox[1] - 2,
                          'gt: difficult: {:d}, detected: {:d}, truncated: {:d}'.format(int(difficult), int(detected),
                                                                                        int(truncated)),
                          bbox=dict(facecolor='blue', alpha=0.5),
                          fontsize=8, color='white')
              else:
                  ax.add_patch(
                      plt.Rectangle((bbox[0], bbox[1]),
                                    bbox[2] - bbox[0],
                                    bbox[3] - bbox[1], fill=False, edgecolor='yellow',
                                    linewidth=1.5, linestyle='solid')
                  )
                  ax.text(bbox[0], bbox[1] - 2,
                          'gt: difficult: {:d}, detected: {:d}, truncated: {:d}'.format(int(difficult), int(detected), int(truncated)),
                          bbox=dict(facecolor='yellow', alpha=0.5),
                          fontsize=8, color='white')

      if det_num != 0:
          for j in range(det_num):
              bbox = det_results[img_name]['det']['bbox'][j]
              score = det_results[img_name]['det']['score'][j]
              result_info = det_results[img_name]['det']['result_info'][j]
              overlap = det_results[img_name]['det']['overlap'][j]
              if result_info == 'correct':
                  ax.add_patch(
                      plt.Rectangle((bbox[0], bbox[1]),
                                    bbox[2] - bbox[0],
                                    bbox[3] - bbox[1], fill=False, edgecolor='green',
                                    linewidth=1.5, linestyle='dashed'))
                  ax.text(bbox[0], bbox[3],
                          '{:s}, score={:f}, overlap={:f}'.format(result_info, score, overlap),
                          bbox=dict(facecolor='green', alpha=0.5),
                          fontsize=8, color='white')
              else:
                  ax.add_patch(
                      plt.Rectangle((bbox[0], bbox[1]),
                                    bbox[2] - bbox[0],
                                    bbox[3] - bbox[1], fill=False, edgecolor='red',
                                    linewidth=1.5, linestyle='dashed')
                  )
                  ax.text(bbox[0], bbox[3],
                          '{:s}, score={:f}, overlap={:f}'.format(result_info, score, overlap),
                          bbox=dict(facecolor='red', alpha=0.5),
                          fontsize=8, color='white')


      ax.set_title('img_name: {:s}, class: {:s}'.format(img_name, classname),
                   fontsize=14)
      plt.axis('off')
      plt.tight_layout()

      if gt_num == 0:
        plt.savefig('/data/zhbli/VOCdevkit/results/VOC2007/vgg16_faster-rcnn/{:s}/no_gt_fp/{:s}.jpg'.format(classname, img_name))
      elif det_num != 0 and det_results[img_name]['det']['result_info'].count('correct')==len(det_results[img_name]['det']['result_info']) \
              and det_results[img_name]['gt']['detected'].count(1) == len(det_results[img_name]['gt']['detected']):
        plt.savefig('/data/zhbli/VOCdevkit/results/VOC2007/vgg16_faster-rcnn/{:s}/correct/{:s}.jpg'.format(classname, img_name))
      else:
        plt.savefig('/data/zhbli/VOCdevkit/results/VOC2007/vgg16_faster-rcnn/{:s}/other/{:s}.jpg'.format(classname, img_name))
  # v3.0

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap
