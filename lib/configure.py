# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Fast R-CNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from configure import cfg
cfg = __C

__C.WSL = False

# __C.PSEUDO_PATH = 'output/cpg/voc_2007_trainval/detections.pkl'
# __C.PSEUDO_PATH = 'output/vgg16_fast_rcnn_wsl_0313/voc_2007_trainval/vgg16_fast_rcnn_iter_40000/detections_o.pkl'

__C.GENERATE_ROI = False
__C.LEFT = 1.0


def get_vis_dir(imdb, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(
        osp.join(__C.ROOT_DIR, 'vis', __C.EXP_DIR, imdb.name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    file_path = 'tmp'
    if os.path.islink(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        import shutil
        shutil.rmtree(file_path)
    else:
        # It is a file
        os.remove(file_path)

    os.symlink(outdir, file_path)
    return outdir


def get_output_dir(imdb, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(
        osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if not isinstance(a, edict):
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if not isinstance(v, old_type):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if isinstance(v, edict):
            try:
                _merge_a_into_b(a[k], b[k])
            except BaseException:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except BaseException:            # handle the case when v is a string literal
            value = v
        assert isinstance(value, type(d[subkey])), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value


def cfg_basic_generation(cfg_origin):
    _clone_a_into_b(cfg_origin, __C)


def _clone_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if not isinstance(a, edict):
        return

    for k, v in a.iteritems():
        # recursively merge dicts
        if isinstance(v, edict):
            b[k] = edict()
            _clone_a_into_b(a[k], b[k])
        else:
            b[k] = v
