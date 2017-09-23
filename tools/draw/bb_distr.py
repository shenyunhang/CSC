#!/usr/bin/env python

import _init_paths
from wsl.train import get_training_roidb, train_net
from wsl.config import cfg_wsl
from configure import cfg, cfg_basic_generation, cfg_from_file, cfg_from_list
from configure import get_output_dir, get_vis_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import cv2
import math


def gray2red(f):
    return (0, 0, f * 255)


def gray2jet(f):
    # plot short rainbow RGB
    a = f / 0.25  # invert and group
    X = math.floor(a)  # this is the integer part
    Y = math.floor(255 * (a - X))  # fractional part from 0 to 255
    Z = math.floor(128 * (a - X))  # fractional part from 0 to 128

    if X == 0:
        r = 0
        g = Y
        b = 128 - Z
    elif X == 1:
        r = Y
        g = 255
        b = 0
    elif X == 2:
        r = 255
        g = 255 - Z
        b = 0
    elif X == 3:
        r = 255
        g = 128 - Z
        b = 0
    elif X == 4:
        r = 255
        g = 0
        b = 0
    #opencv is bgr, not rgb
    return (b, g, r)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str)
    parser.add_argument(
        '--imdb',
        dest='imdb_name',
        help='dataset to train on',
        default='voc_2007_trainval',
        type=str)
    parser.add_argument(
        '--rand',
        dest='randomize',
        help='randomize (do not use a fixed seed)',
        action='store_true')
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='set config keys',
        default=None,
        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(imdb_names):
    # treat as only one dataset
    imdb = get_imdb(imdb_names)

    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    roidb = get_training_roidb(imdb)
    return imdb, roidb

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    cfg_basic_generation(cfg_wsl)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    imdb, roidb = combined_roidb(args.imdb_name)
    print '{:d} roidb entries'.format(len(roidb))

    max_try = 1000
    max_show_per = 30
    max_show = 30
    for i in xrange(len(roidb)):
        print roidb[i]

        img = cv2.imread(roidb[i]['image'])
        height_img, width_img, _ = img.shape

        rois = roidb[i]['boxes']
        bboxes = []
        ovs = []
        for j in xrange(len(rois)):
            if j >= max_show:
                break
            roi = rois[j]
            x1 = int(roi[0])
            y1 = int(roi[1])
            x2 = int(roi[2])
            y2 = int(roi[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            suc_try = 0
            all_try = 0
            while all_try >= max_try or suc_try >= max_show_per:
                # cx = np.random.normal(1.0 * (x1 + x2) / 2, width_img)
                # cy = np.random.normal(1.0 * (y1 + y2) / 2, height_img)
                cx = np.random.normal(1.0 * (x1 + x2) / 2, 1)
                cy = np.random.normal(1.0 * (y1 + y2) / 2, 1)

                print 1.0 * (x1 + x2) / 2, 1.0 * (y1 + y2) / 2, cx, cy

                # h = np.random.normal(1.0 * (y2 - y1), height_img)
                # w = np.random.normal(1.0 * (x2 - x1), width_img)
                h = np.random.normal(1.0 * (y2 - y1), (y2 - y1))
                w = np.random.normal(1.0 * (x2 - x1), (x2 - x1))
                # w = h / (y2 - y1) * (x2 - x1)

                xx1 = max(0, int(cx - w / 2))
                yy1 = max(0, int(cy - h / 2))
                xx2 = min(width_img, int(cx + w / 2))
                yy2 = min(height_img, int(cy + h / 2))

                xxx1 = max(x1, xx1)
                xxx2 = min(x2, xx2)
                yyy1 = max(y1, yy1)
                yyy2 = min(y2, yy2)
                if xxx1 > xxx2 or yyy1 > yyy2:
                    ov = -1
                else:
                    ins = (xxx2 - xxx1) * (yyy2 - yyy1)
                    ov = 1.0 * ins / ((x2 - x1) * (y2 - y1) + (xx2 - xx1) *
                                      (yy2 - yy1) - ins)
                print ov
                if ov > 0.0:
                    # cv2.rectangle(img, (xx1, yy1), (xx2, yy2), (0, 255, 0), 1)
                    suc_try = suc_try + 1
                    bboxes.append([xx1, yy1, xx2, yy2])
                    ovs.append(ov)
                # else:
                # cv2.rectangle(img, (xx1, yy1), (xx2, yy2), (255, 0, 0), 1)
                all_try = all_try + 1

        idx_sort = sorted(range(len(ovs)), key=lambda k: ovs[k])
        for j in xrange(len(idx_sort)):
            idx = idx_sort[j]
            x1 = int(bboxes[idx][0])
            y1 = int(bboxes[idx][1])
            x2 = int(bboxes[idx][2])
            y2 = int(bboxes[idx][3])
            cv2.rectangle(img, (x1, y1), (x2, y2), gray2jet(ovs[idx]), 1)

        for j in xrange(len(rois)):
            if j >= max_show:
                break
            roi = rois[j]
            x1 = int(roi[0])
            y1 = int(roi[1])
            x2 = int(roi[2])
            y2 = int(roi[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # cv2.imshow('img', img)
        # cv2.waitKey()

        cv2.imwrite('tmp/{}.jpg'.format(i), img)
