# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval, voc_eval_corloc, voc_eval_visualization
from configure import cfg
import PIL


class pascal_voc(imdb):

    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        if cfg.WSL:
            # remove the _background_ class
            self._classes = self._classes[1:]
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()

        # remove some training data
        if 'test' not in self._name and cfg.LEFT < 1.0:
            self._remove_ims()
        else:
            print 'using all image'

        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'
        self._comp_id_cls = 'comp1'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 20}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

        # test split of PASCAL VOC >2007
        if 'test' in self._name and int(self._year) > 2007:
            return

        self._gt_classes = {ix: self._load_pascal_classes_annotation(
            ix) for ix in self._image_index}

    def _remove_ims(self):
        self._image_index_old = self._image_index
        self._image_index = []
        self._image_left = []
        for index in self._image_index_old:
            r = np.random.random()
            if r <= cfg.LEFT:
                self._image_index.append(index)
                self._image_left.append(1)
            else:
                self._image_left.append(0)

        print 'original image number: ', len(self._image_index_old)
        print 'left image number:', len(self._image_index)

    def image_classes_at(self, i):
        """
        Return the gt class to image i in the image sequence.
        """
        return self.image_classes_from_index(self._image_index[i])

    def image_classes_from_index(self, index):
        """
        Construct an image classes from the image's "index" identifier.
        """
        return self._gt_classes[index]

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # TODO(YH): current we do not use cache, due to num_classes is different
        # cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # if os.path.exists(cache_file):
        # with open(cache_file, 'rb') as fid:
        # roidb = cPickle.load(fid)
        # print '{} gt roidb loaded from {}'.format(self.name, cache_file)
        # return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        # with open(cache_file, 'wb') as fid:
        # cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        # print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def pseudo_gt_roidb(self):
        """
        Return the database of pseudo ground-truth regions of interest from detect result.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        # detection.pkl is 0-based indices
        # the VOC result file is 1-based indices

        cache_file = cfg.PSEUDO_PATH
        assert os.path.exists(cache_file)
        with open(cache_file, 'rb') as fid:
            roidb = cPickle.load(fid)
        print '{} pseudo gt roidb loaded from {}'.format(self.name, cache_file)

        assert len(roidb[0]) == self.num_images
        if len(roidb) == self.num_classes:
            cls_offset = 0
        elif len(roidb) + 1 == self.num_classes:
            cls_offset = -1
        else:
            raise Exception('Unknown mode.')

        gt_roidb = []
        for im_i, ix in enumerate(self._image_index):
            if im_i % 1000 == 0:
                print '{:d} / {:d}'.format(im_i + 1, self.num_images)
            num_objs = 0

            # when cfg.WSL = False, background class is in.
            # detection.pkl only has 20 classes
            # fast_rcnn need 21 classes
            for cls in range(1, self.num_classes):
                # TODO(YH): we need threshold the pseudo label
                # filter the pseudo label
                # self._gt_class has 21 classes
                if self._gt_classes[ix][cls] == 0:
                    continue
                dets = roidb[cls + cls_offset][im_i]
                # num_objs += bbox.shape[0]
                # TODO(YH): keep only one box
                if dets.shape[0] > 0:
                    num_objs += 1

            if num_objs == 0:
                continue

            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            # "Seg" area for pascal is just the box area
            seg_areas = np.zeros((num_objs), dtype=np.float32)

            obj_i = 0
            for cls in range(1, self.num_classes):
                # filter the pseudo label
                # self._gt_class has 21 classes
                if self._gt_classes[ix][cls] == 0:
                    continue
                dets = roidb[cls + cls_offset][im_i]
                if dets.shape[0] <= 0:
                    continue

                max_score = 0
                for i in range(dets.shape[0]):
                    det = dets[i]
                    x1 = det[0]
                    y1 = det[1]
                    x2 = det[2]
                    y2 = det[3]

                    score = det[4]
                    if score <= max_score:
                        continue
                    max_score = score

                    assert x2 >= x1
                    assert y2 >= y1
                    boxes[obj_i, :] = [x1, y1, x2, y2]
                    gt_classes[obj_i] = cls
                    overlaps[obj_i, cls] = 1.0
                    seg_areas[obj_i] = (x2 - x1 + 1) * (y2 - y1 + 1)

                obj_i += 1

            assert obj_i == num_objs

            gt_roidb.append(
                {'boxes': boxes,
                 'box_scores': np.ones((num_objs, 1), dtype=np.float32),
                 'gt_classes': gt_classes,
                 'gt_overlaps': overlaps,
                 'flipped': False,
                 'seg_areas': seg_areas}
            )

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # TODO(YH): current we do not use cache, due to num_classes is different
        # cache_file = os.path.join(self.cache_path,
        # self.name + '_selective_search_roidb.pkl')

        # if os.path.exists(cache_file):
        # with open(cache_file, 'rb') as fid:
        # roidb = cPickle.load(fid)
        # print '{} ss roidb loaded from {}'.format(self.name, cache_file)
        # return roidb

        if cfg.WSL and not cfg.USE_PSEUDO:
            # WSL train and test
            roidb = self._load_selective_search_roidb(None)
        elif not cfg.WSL and cfg.USE_PSEUDO and 'trainval' in self.name:
            # WSL fast rcnn train
            pseudo_gt_roidb = self.pseudo_gt_roidb()
            ss_roidb = self._load_selective_search_roidb(pseudo_gt_roidb)
            roidb = imdb.merge_roidbs(pseudo_gt_roidb, ss_roidb)
        elif not cfg.WSL and cfg.USE_PSEUDO and 'test' in self.name:
            # WSL fast rcnn test
            roidb = self._load_selective_search_roidb(None)
        elif not cfg.WSL and not cfg.USE_PSEUDO:
            # Fast rcnn train and test
            if (int(self._year) == 2007 or self._image_set != 'test'):
                gt_roidb = self.gt_roidb()
                ss_roidb = self._load_selective_search_roidb(gt_roidb)
                roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
            else:
                roidb = self._load_selective_search_roidb(None)
        else:
            raise Exception('Not implement mode.')

        # with open(cache_file, 'wb') as fid:
            # cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        # print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def edge_boxes_roidb(self):
        """
        Return the database of edge boxes regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # TODO(YH): current we do not use cache, due to num_classes is different
        # cache_file = os.path.join(self.cache_path,
        # self.name + '_edge_boxes_roidb.pkl')

        # if os.path.exists(cache_file):
        # with open(cache_file, 'rb') as fid:
        # roidb = cPickle.load(fid)
        # print '{} ss roidb loaded from {}'.format(self.name, cache_file)
        # return roidb

        if cfg.WSL and not cfg.USE_PSEUDO:
            # WSL train and test
            roidb = self._load_edge_boxes_roidb(None)
        elif not cfg.WSL and cfg.USE_PSEUDO and 'trainval' in self.name:
            # WSL fast rcnn train
            pseudo_gt_roidb = self.pseudo_gt_roidb()
            eb_roidb = self._load_edge_boxes_roidb(pseudo_gt_roidb)
            roidb = imdb.merge_roidbs(pseudo_gt_roidb, eb_roidb)
        elif not cfg.WSL and cfg.USE_PSEUDO and 'test' in self.name:
            # WSL fast rcnn test
            roidb = self._load_edge_boxes_roidb(None)
        elif not cfg.WSL and not cfg.USE_PSEUDO:
            # Fast rcnn train and test
            if (int(self._year) == 2007 or self._image_set != 'test'):
                gt_roidb = self.gt_roidb()
                eb_roidb = self._load_edge_boxes_roidb(gt_roidb)
                roidb = imdb.merge_roidbs(gt_roidb, eb_roidb)
            else:
                roidb = self._load_edge_boxes_roidb(None)
        else:
            raise Exception('Not implement mode.')

        # with open(cache_file, 'wb') as fid:
            # cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        # print 'wrote eb roidb to {}'.format(cache_file)

        return roidb

    def _load_edge_boxes_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'EdgeBoxes70_' +
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Edge boxes data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['bbs'].ravel()
        # print raw_data.shape

        box_list = []
        score_list = []
        total_roi = 0
        up_1024 = 0
        up_2048 = 0
        up_3072 = 0
        up_4096 = 0

        for i in xrange(raw_data.shape[0]):
            if i % 1000 == 0:
                print '{:d} / {:d}'.format(i + 1, len(self._image_index))

            # bbs        - [nx5] array containing proposal bbs [x y w h score]
            x1 = raw_data[i][:, 0] - 1
            y1 = raw_data[i][:, 1] - 1
            x2 = raw_data[i][:, 2] + x1
            y2 = raw_data[i][:, 3] + y1
            scores = raw_data[i][:, 4]

            boxes = np.squeeze(np.dstack((x1, y1, x2, y2)))
            scores = np.squeeze(scores)
            scores = np.reshape(scores, (boxes.shape[0], 1))

            # print raw_data[i].shape
            # print x1.shape, y1.shape, x2.shape, y2.shape
            # print boxes.shape
            # print x1[0], y1[0], x2[0], y2[0]
            # print boxes[0, :]
            # print x1[-1], y1[-1], x2[-1], y2[-1]
            # print boxes[-1, :]
            # print scores.shape

            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            scores = scores[keep]

            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            scores = scores[keep]

            total_roi += boxes.shape[0]
            box_list.append(boxes)
            score_list.append(scores)

            assert boxes.shape[0] == scores.shape[
                0], 'box num({}) should equal score num({})'.format(boxes.shape, scores.shape)

            total_roi += boxes.shape[0]
            if boxes.shape[0] > 1024:
                up_1024 += 1
            if boxes.shape[0] > 2048:
                up_2048 += 1
            if boxes.shape[0] > 3072:
                up_3072 += 1
            if boxes.shape[0] > 4096:
                up_4096 += 1

        print 'total_roi: ', total_roi, ' ave roi: ', total_roi / len(box_list)
        print 'up_1024: ', up_1024
        print 'up_2048: ', up_2048
        print 'up_3072: ', up_3072
        print 'up_4096: ', up_4096
        return self.create_roidb_from_box_list(box_list, gt_roidb, score_list)

    def _load_edge_boxes_roidb_despreate(self):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'edge_boxes_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Edge boxes data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()
        score_data = sio.loadmat(filename)['boxScores'].ravel()

        box_list = []
        score_list = []
        total_roi = 0
        for i in xrange(raw_data.shape[0]):
            # boxes in eb are in the form [y1 x1 y2 x2]
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            scores = score_data[i]

            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            scores = scores[keep]

            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            scores = scores[keep]

            total_roi += boxes.shape[0]
            box_list.append(boxes)
            score_list.append(scores)

        print 'total_roi: ', total_roi, ' ave roi: ', total_roi / i
        return self.create_roidb_from_box_list(box_list, None, score_list)

    def mcg_roidb(self):
        """
        Return the database of Multiscale Combinatorial Grouping regions of interest.
        """
        # TODO(YH): current we do not use cache, due to num_classes is different
        # cache_file = os.path.join(self.cache_path,
        # self.name + '_mcg_roidb.pkl')
        # if os.path.exists(cache_file):
        # with open(cache_file, 'rb') as fid:
        # roidb = cPickle.load(fid)
        # print '{} mcg roidb loaded from {}'.format(self.name, cache_file)
        # return roidb

        if cfg.WSL and not cfg.USE_PSEUDO:
            # WSL train and test
            roidb = self._load_mcg_roidb(None)
        elif not cfg.WSL and cfg.USE_PSEUDO and 'trainval' in self.name:
            # WSL fast rcnn train
            pseudo_gt_roidb = self.pseudo_gt_roidb()
            mcg_roidb = self._load_mcg_roidb(pseudo_gt_roidb)
            roidb = imdb.merge_roidbs(pseudo_gt_roidb, mcg_roidb)
        elif not cfg.WSL and cfg.USE_PSEUDO and 'test' in self.name:
            # WSL fast rcnn test
            roidb = self._load_mcg_roidb(None)
        elif not cfg.WSL and not cfg.USE_PSEUDO:
            # Fast rcnn train and test
            if (int(self._year) == 2007 or self._image_set != 'test'):
                gt_roidb = self.gt_roidb()
                mcg_roidb = self._load_mcg_roidb(gt_roidb)
                roidb = imdb.merge_roidbs(gt_roidb, mcg_roidb)
            else:
                roidb = self._load_mcg_roidb(None)
        else:
            raise Exception('Not implement mode.')

        if cfg.GENERATE_ROI:
            gt_roidb = self.gt_roidb()
            roidb = self._general_roidb(gt_roidb)

        # with open(cache_file, 'wb') as fid:
            # cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        # print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _general_roidb(self, gt_roidb):
        roidb = []
        num_objs1 = 40
        num_objs2 = 40
        num_objs = num_objs1 + num_objs2 + 1
        step = 6
        for i, index in enumerate(self.image_index):
            size = PIL.Image.open(self.image_path_at(i)).size
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            box_scores = np.zeros((num_objs, 1), dtype=np.float32)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            # "Seg" area for pascal is just the box area
            seg_areas = np.zeros((num_objs), dtype=np.float32)

            gt_boxes = gt_roidb[i]['boxes']
            x1 = gt_boxes[0][0]
            y1 = gt_boxes[0][1]
            x2 = gt_boxes[0][2]
            y2 = gt_boxes[0][3]

            for ix in xrange(num_objs1):
                x1 += step
                y1 += step
                x2 -= step
                y2 -= step
                if x1 + step > x2 or y1 + step > y2:
                    continue
                boxes[ix, :] = [x1, y1, x2, y2]
                box_scores[ix] = 1.0
                gt_classes[ix] = 0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

            x1 = gt_boxes[0][0]
            y1 = gt_boxes[0][1]
            x2 = gt_boxes[0][2]
            y2 = gt_boxes[0][3]
            for ix in xrange(num_objs1, num_objs - 1):
                x1 -= step
                y1 -= step
                x2 += step
                y2 += step
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 >= size[0]:
                    x2 = size[0] - 1
                if y2 >= size[1]:
                    y2 = size[1] - 1

                boxes[ix, :] = [x1, y1, x2, y2]
                box_scores[ix] = 1.0
                gt_classes[ix] = 0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

            x1 = gt_boxes[0][0]
            y1 = gt_boxes[0][1]
            x2 = gt_boxes[0][2]
            y2 = gt_boxes[0][3]
            ix = num_objs - 1
            boxes[ix, :] = [x1, y1, x2, y2]
            box_scores[ix] = 1.0
            gt_classes[ix] = 0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

            roidb.append({'boxes': boxes,
                          'box_scores': box_scores,
                          'gt_classes': gt_classes,
                          'gt_overlaps': overlaps,
                          'flipped': False,
                          'seg_areas': seg_areas}
                         )
        return roidb

    def _load_mcg_roidb(self, gt_roidb):
        box_list = []
        score_list = []
        total_roi = 0
        up_1024 = 0
        up_2048 = 0
        up_3072 = 0
        up_4096 = 0

        for i, index in enumerate(self._image_index):
            if i % 1000 == 0:
                print '{:d} / {:d}'.format(i + 1, len(self._image_index))

            box_file = os.path.join(
                cfg.DATA_DIR, 'MCG-Pascal-Main_trainvaltest_{}-boxes'.format(self._year), '{}.mat'.format(index))

            raw_data = sio.loadmat(box_file)['boxes']
            score_data = sio.loadmat(box_file)['scores']

            boxes = np.maximum(raw_data - 1, 0).astype(np.uint16)
            scores = score_data.astype(np.float)

            # Boxes from the MCG website are in (y1, x1, y2, x2) order
            boxes = boxes[:, (1, 0, 3, 2)]

            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            scores = scores[keep]

            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            scores = scores[keep]

            # sort by confidence
            sorted_ind = np.argsort(-scores.flatten())
            scores = scores[sorted_ind, :]
            boxes = boxes[sorted_ind, :]

            assert boxes.shape[0] == scores.shape[
                0], 'box num({}) should equal score num({})'.format(boxes.shape, scores.shape)

            total_roi += boxes.shape[0]
            if boxes.shape[0] > 1024:
                up_1024 += 1
            if boxes.shape[0] > 2048:
                up_2048 += 1
            if boxes.shape[0] > 3072:
                up_3072 += 1
            if boxes.shape[0] > 4096:
                up_4096 += 1

            box_list.append(boxes)
            score_list.append(scores)

        print 'total_roi: ', total_roi, ' ave roi: ', total_roi / len(box_list)
        print 'up_1024: ', up_1024
        print 'up_2048: ', up_2048
        print 'up_3072: ', up_3072
        print 'up_4096: ', up_4096
        return self.create_roidb_from_box_list(box_list, gt_roidb, score_list)

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        total_roi = 0
        up_1024 = 0
        up_2048 = 0
        up_3072 = 0
        up_4096 = 0
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            total_roi += boxes.shape[0]
            box_list.append(boxes)

            if boxes.shape[0] > 1024:
                up_1024 += 1
            if boxes.shape[0] > 2048:
                up_2048 += 1
            if boxes.shape[0] > 3072:
                up_3072 += 1
            if boxes.shape[0] > 4096:
                up_4096 += 1

        print 'total_roi: ', total_roi, ' ave roi: ', total_roi / len(box_list)
        print 'up_1024: ', up_1024
        print 'up_2048: ', up_2048
        print 'up_3072: ', up_3072
        print 'up_4096: ', up_4096

        print 'total_roi: ', total_roi, ' ave roi: ', total_roi / i
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_classes_annotation(self, index):
        """
        Load image and classes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        gt_classes = np.zeros(self.num_classes, dtype=np.int32)

        # Load object class into a data frame.
        for ix, obj in enumerate(objs):
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            gt_classes[cls] = 1

        # return {'gt_classes': gt_classes}
        return gt_classes

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        box_scores = np.zeros((num_objs, 1), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            box_scores[ix] = 1.0
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'box_scores': box_scores,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def set_salt(self, salt):
        self._salt = salt

    def _get_comp_id_cls(self):
        comp_id = (self._comp_id_cls + '_' + self._salt if self.config['use_salt']
                   else self._comp_id_cls)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + \
            self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _get_voc_results_file_template_cls(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id_cls() + '_cls_' + \
            self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file_cls(self, all_scores):
        if all_scores is None:
            return
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template_cls().format(cls)
            print 'Writing {} VOC results file \t{}'.format(cls, os.path.relpath(filename))
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    score = all_scores[cls_ind][im_ind]
                    if score == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    f.write('{:s} {:.8f}\n'.
                            format(index, score))

    def _write_voc_results_file(self, all_boxes):
        if all_boxes is None:
            return
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            print 'Writing {} VOC results file \t{}'.format(cls, os.path.relpath(filename))
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.8f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f} \tfilename: {}'.format(
                cls, ap, os.path.relpath(filename)))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.4f}'.format(ap))
        print('{:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

        print('Results:')
        for ap in aps:
            print('{:.2f}\t'.format(ap * 100.0)),
        print('{:.2f}'.format(np.mean(aps) * 100.0))

        cfg.TEST.MAP = np.mean(aps)

    def _do_python_eval_corloc(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        corlocs = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            corloc = voc_eval_corloc(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric)
            corlocs += [corloc]
            print('corloc for {} = {:.4f} \tfilename: {}'.format(
                cls, corloc, os.path.relpath(filename)))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'corloc': corloc}, f)
        print('Mean corloc = {:.4f}'.format(np.mean(corlocs)))
        print('~~~~~~~~')
        print('Results:')
        for corloc in corlocs:
            print('{:.4f}'.format(corloc))
        print('{:.4f}'.format(np.mean(corlocs)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

        print('Results:')
        for corloc in corlocs:
            print('{:.2f}\t'.format(corloc * 100.0)),
        print('{:.2f}'.format(np.mean(corlocs) * 100.0))

        cfg.TEST.MAP = np.mean(corlocs)

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir, all_scores=None):
        self._write_voc_results_file(all_boxes)
        self._write_voc_results_file_cls(all_scores)

        if 'trainval' in self._name:
            self._do_python_eval_corloc(output_dir)
        else:
            self._do_python_eval(output_dir)
            if self.config['matlab_eval']:
                self._do_matlab_eval(output_dir)

        # if self.config['cleanup']:
        if False:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def visualization_detection(self, output_dir='vis'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  '{}' + self._image_ext)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            output_sub_dir = os.path.join(output_dir, cls)
            # output_sub_dir = output_dir
            if not os.path.isdir(output_sub_dir):
                os.mkdir(output_sub_dir)
            filename = self._get_voc_results_file_template().format(cls)
            voc_eval_visualization(
                filename, annopath, imagesetfile, cls, cachedir, image_path, output_sub_dir)


if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed
    embed()
