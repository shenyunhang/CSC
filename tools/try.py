#!/usr/bin/env python

import _init_paths
from fast_rcnn.train import get_training_roidb
from wsl.config import cfg_wsl
from configure import cfg, cfg_basic_generation, cfg_from_file, cfg_from_list
import argparse
import sys
import pprint
from datasets.factory import get_imdb
import roi_data_layer.roidb as rdl_roidb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--snapshot', dest='snapshot_state',
                        help='initialize with snapshot solverstate',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')

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

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    # print 'Appending horizontally-flipped training examples...'
    # imdb.append_flipped_images()
    # print 'done'

    # print 'Preparing training data...'
    # rdl_roidb.prepare_roidb(imdb)
    # print 'done'

    # our
    # imdb.set_salt('60f388c4-d450-4e78-88ed-5d1a669eb9f0')
    # baseline
    imdb.set_salt('ade24af4-a439-4c21-a678-85102287437c')

    imdb.evaluate_detections(None, 'output', None)

    # imdb.visualization_detection('vis/draw_our')
    imdb.visualization_detection('vis/draw_wsddn')
