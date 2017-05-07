#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, glob, cv2
import argparse

CLASSES = ('__background__',
             'person','tennis court',
		'dog','horse','axe',
		'bicycle','brush','coffee cup',
		'hose','clothes iron','lawn mower',
		'tennis racquet','toothbrush','vacuum cleaner')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'zfim': ('ZF',
                'ZF_faster_rcnn_imagenet_final.caffemodel')}

def detection(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    basename = os.path.basename(image_name)
    outputfile = '{}.mat'.format(os.path.join(args.save_dir, basename))
    if os.path.exists(outputfile):
        return

    # Load the demo image
    im_file = os.path.join(image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im)
    
    boxsave = np.vstack((np.mean(boxes[:,4::4],axis=1),np.mean(boxes[:,5::4],axis=1),np.mean(boxes[:,6::4],axis=1),np.mean(boxes[:,7::4],axis=1))).T
    zs = scores[:,1:]
    
    if args.debug:
        print 'Saving scores and BBs to mat file'


    #print ('{}.mat'.format(os.path.join(args.save_dir, basename)))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    data = dict((ocls.replace(' ','_'), []) for ocls in CLASSES[1:])
    for cls_ind, ocls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        data[ocls.replace(' ','_')] = dets

    sio.savemat(outputfile, {'boxes':boxsave,'zs':zs,'data':data})

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_dir')
    parser.add_argument('save_dir')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    #parser.add_argument('--savemat', action='store_true', help='Store scores and BBs to mat files')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    images = sorted(glob.glob(os.path.join(args.frames_dir,'*')))
    print ("Processing {}: {} files... ".format(args.frames_dir, len(images))),
    sys.stdout.flush

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    timer = Timer()
    timer.tic()
    for image in images:
        if args.debug:
            print ("Processing file {}".format(image))
        detection(net, image)
    timer.toc()
    print "{:.2f} min, {:.2f} fps".format((timer.total_time) / 60., 1. * len(images) / (timer.total_time))
