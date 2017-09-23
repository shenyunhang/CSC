#!/usr/bin/env python

import _init_paths
from wsl.config import cfg_wsl
from configure import cfg, cfg_basic_generation, cfg_from_file, cfg_from_list
import argparse
import sys
import pprint
from datasets.factory import get_imdb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Eval a result')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str)
    parser.add_argument(
        '--imdb',
        dest='imdb_name',
        help='dataset to eval on',
        default='voc_2007_trainval',
        type=str)
    parser.add_argument(
        '--salt', dest='salt', help='salt to eval on', default=None, type=str)
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
    parser.add_argument(
        '--comp',
        dest='comp_mode',
        help='competition mode',
        action='store_true')

    # if len(sys.argv) == 1:
    # parser.print_help()
    # sys.exit(1)

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

    print('Using config:')
    pprint.pprint(cfg)

    imdb = get_imdb(args.imdb_name)

    imdb.competition_mode(args.comp_mode)
    if not args.comp_mode and args.salt is not None:
        imdb.set_salt(args.salt)

    imdb.evaluate_detections(None, 'output', None)

    # imdb.visualization_detection('tmp')
