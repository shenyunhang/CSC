import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from cpg.config import cfg_cpg
cfg_ssd = __C

#
# Training options
#

__C.TRAIN = edict()

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600, )

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Images to use per minibatch
# If image per Batch lagerer than 64, blob will exceed INT_MAX.
__C.TRAIN.IMS_PER_BATCH = 2

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

__C.TRAIN.ROIS_PER_IM = 10000

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

__C.TRAIN.USE_DISTORTION = True
__C.TRAIN.brightness_prob = 0.5
__C.TRAIN.brightness_delta = 32
__C.TRAIN.contrast_prob = 0.5
__C.TRAIN.contrast_lower = 0.5
__C.TRAIN.contrast_upper = 1.5
__C.TRAIN.hue_prob = 0.5
__C.TRAIN.hue_delta = 18
__C.TRAIN.saturation_prob = 0.5
__C.TRAIN.saturation_lower = 0.5
__C.TRAIN.saturation_upper = 1.5
__C.TRAIN.exposure_prob = 0.5
__C.TRAIN.exposure_lower = 0.5
__C.TRAIN.exposure_upper = 1.5
__C.TRAIN.random_order_prob = 0.0

__C.TRAIN.USE_EXPAND = True
__C.TRAIN.expand_prob = 0.5
__C.TRAIN.max_expand_ratio = 4.0

__C.TRAIN.USE_SAMPLE = True
__C.TRAIN.batch_sampler = [
    edict({
        'sampler': {
            'min_scale': 1.0,
            'max_scale': 1.0,
            'min_aspect_ratio': 1.0,
            'max_aspect_ratio': 1.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.0,
            'max_jaccard_overlap': 1.0,
        },
        'max_trials': 1,
        'max_sample': 1,
    }),
    edict({
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.1,
            'max_jaccard_overlap': 1.0,
        },
        'max_trials': 50,
        'max_sample': 1,
    }),
    edict({
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.3,
            'max_jaccard_overlap': 1.0,
        },
        'max_trials': 50,
        'max_sample': 1,
    }),
    edict({
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.5,
            'max_jaccard_overlap': 1.0,
        },
        'max_trials': 50,
        'max_sample': 1,
    }),
    edict({
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.7,
            'max_jaccard_overlap': 1.0,
        },
        'max_trials': 50,
        'max_sample': 1,
    }),
    edict({
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.9,
            'max_jaccard_overlap': 1.0,
        },
        'max_trials': 50,
        'max_sample': 1,
    }),
    edict({
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.9,
            'max_jaccard_overlap': 1.0,
        },
        'max_trials': 50,
        'max_sample': 1,
    }),
]

__C.TRAIN.INTERP_MODEL = ['LINEAR', 'AREA', 'NEAREST', 'CUBIC', 'LANCZOS4']

__C.TRAIN.ROI_AU = False
__C.TRAIN.ROI_AU_STEP = 1

__C.TRAIN.OPG_CACHE = False
__C.TRAIN.OPG_CACHE_PATH = 'data/opg_cache/'

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 10000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
__C.TRAIN.USE_PREFETCH = False

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'selective_search'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True

__C.TRAIN.PASS_IM = 0

__C.TRAIN.SHUFFLE = True

__C.TRAIN.PSEUDO_PATH = 'output/vgg16_cpg_0921/voc_2007_trainval/VGG16_2_iter_10/detections_o.pkl'

#
# Testing options
#

__C.TEST = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600, )

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'selective_search'

__C.TEST.ROIS_PER_IM = 10000
__C.TEST.USE_FLIPPED = True
__C.TEST.BBOX = False

# for grid search NMS max_per_image thresh and so on
__C.TEST.CACHE = False
__C.TEST.MAP = 0.0

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1. / 16.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# fast rcnn
# __C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
# VGG 16
__C.PIXEL_MEANS = np.array([[[103.939, 116.779, 123.68]]])
# CaffeNet
# __C.PIXEL_MEANS = np.array([[[104.00, 117.00, 123.00]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Model directory
__C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models', 'pascal_voc'))

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0

__C.OPG_DEBUG = False

__C.CONTEXT = False
__C.CONTEXT_RATIO = 1.8

__C.USE_ROI_SCORE = False

__C.USE_BG = False

__C.SPATIAL_SCALE = 1. / 16.

__C.RESIZE_MODE = 'WARP'


def get_vis_dir(imdb, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'vis', __C.EXP_DIR, imdb.name))
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
