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
#             'person','tennis court',
#		'dog','horse','axe',
#		'bicycle','brush','coffee cup',
#		'hose','clothes iron','lawn mower',
#		'tennis racquet','toothbrush','vacuum cleaner')
                        'tennis court',
			'dog','horse','axe','ball',
			'bicycle','bottle','brush','bucket',
			'coffee cup','garage','glass',
			'gun','hose','clothes iron','lawn mower',
			'net','sword','tennis racquet','toothbrush',
			'tub','vacuum cleaner','washbasin')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'zfim': ('ZF',
                'ZF_faster_rcnn_imagenet_final.caffemodel')}

def vis_detections(im, class_name_all, detsall, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    ax.cla()
    im = im[:, :, (2, 1, 0)]
    ax.imshow(im, aspect='equal')
    for ix, dets in enumerate(detsall):
        inds = np.where(dets[:, -1] >= thresh)[0]
        #print dets
        if len(inds) == 0:
            continue

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
                )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name_all[ix], score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

def detection(net, im, ax):
    """Detect object classes in an image using pre-computed object proposals."""
    '''
    basename = os.path.basename(image_name)
    outputfile = '{}.mat'.format(os.path.join(args.save_dir, basename))
    if os.path.exists(outputfile):
        return

    # Load the demo image
    im_file = os.path.join(image_name)
    im = cv2.imread(im_file)
    '''
    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im)

    #print ('{}.mat'.format(os.path.join(args.save_dir, basename)))

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    detsall = []
    oclsall = []
    data = dict((ocls.replace(' ','_'), []) for ocls in CLASSES[1:])
    for cls_ind, ocls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        detsall.append(dets)
        oclsall.append(ocls)

    vis_detections(im, oclsall, detsall, ax, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
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

    sys.stdout.flush

    print 'Processing video {}'.format(os.path.basename(args.vid_file))
    cap = cv2.VideoCapture(args.vid_file)


    fig, ax = plt.subplots(figsize=(12, 12))

    timer = Timer()
    timer.tic()
    while(cap.isOpened()):
        ret, image = cap.read()
        detection(net, image, ax)
    fig.close()
    cap.release()
    timer.toc()
    print "{:.2f} min, {:.2f} fps".format((timer.total_time) / 60., 1. * len(images) / (timer.total_time))
