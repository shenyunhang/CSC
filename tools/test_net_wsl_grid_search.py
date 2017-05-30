#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from wsl.test import test_net, test_net_cache, test_net_bbox
from wsl.config import cfg_wsl
from configure import cfg, cfg_basic_generation, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time
import os
import sys
import numpy as np
import csv


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    cfg_basic_generation(cfg_wsl)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not os.path.exists(args.caffemodel) and args.wait:
        while not os.path.exists(args.caffemodel) and args.wait:
            current_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
            print('{}: waiting for {} to exist...'.format(
                current_time, args.caffemodel))
            time.sleep(60 * 10)
        time.sleep(60 * 5)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    result = []
    nmses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    max_per_images = [1, 10, 100, 1000, 10000]

    for nms in nmses:
        cfg.TEST.NMS = nms
        for thresh in threshs:
            for max_per_image in max_per_images:
                print '----------------------------------------------------'
                print nms, thresh, max_per_image
                test_net_cache(
                    net, imdb, max_per_image=max_per_image, thresh=thresh)
                result.append([nms, thresh, max_per_image, cfg.TEST.MAP])
    print result
    f = open('grid_search.csv', 'wb')
    wr = csv.writer(f, dialect='excel')
    wr.writerows(result)
