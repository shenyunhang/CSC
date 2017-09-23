# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as npr
import cv2
import os
from configure import cfg
from utils.blob import im_list_to_blob
import datasets.ds_utils
import utils.im_transforms


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)

    processed_ims = []
    # Now, build the region of interest and label blobs
    roi_blob = np.zeros((0, 5), dtype=np.float32)
    roi_context_blob = np.zeros((0, 9), dtype=np.float32)
    roi_frame_blob = np.zeros((0, 9), dtype=np.float32)
    roi_score_blob = np.zeros((0, 1), dtype=np.float32)
    roi_num_blob = np.zeros((0, 1), dtype=np.float32)
    label_blob = np.zeros((0, num_classes), dtype=np.float32)
    opg_filter_blob = np.zeros((0, num_classes), dtype=np.float32)
    opg_io_blob = np.zeros((0, 1), dtype=np.float32)
    for i_im in xrange(num_images):
        # 处理图像
        img = cv2.imread(roidb[i_im]['image'])
        if roidb[i_im]['flipped']:
            img = img[:, ::-1, :]

        # 获得ROI
        # x1 y1 x2 y2
        roi = roidb[i_im]['boxes'].astype(np.float32)
        if cfg.USE_ROI_SCORE:
            roi_score = roidb[i_im]['box_scores']
        else:
            roi_score = np.zeros((roi.shape[0], 1), dtype=np.float32)

        # Check RoI
        datasets.ds_utils.validate_boxes(
            roi, width=img.shape[1], height=img.shape[0])

        #-------------------------------------------------------------
        # 处理ROI
        roi_per_this_image = np.minimum(cfg.TRAIN.ROIS_PER_IM, roi.shape[0])
        roi = roi[:roi_per_this_image, :]
        roi_score = roi_score[:roi_per_this_image]

        # vis(img, roi, show_name='origin')

        # 处理标签
        img_label = roidb[i_im]['gt_classes']
        if cfg.USE_BG:
            img_label = np.hstack((img_label, [1.0]))
        label_blob = np.vstack((label_blob, img_label))

        #-------------------------------------------------------------
        if cfg.TRAIN.USE_DISTORTION:
            img = utils.im_transforms.ApplyDistort(img)
        elif cfg.TRAIN.USE_DISTORTION_OLD:
            img = utils.im_transforms.ApplyDistort_old(img)
        # vis(img, roi, show_name='distortion')

        #-------------------------------------------------------------
        # expand_bbox is define as RoIs with form (x1,y1,x2,y2)
        if cfg.TRAIN.USE_EXPAND:
            img, expand_bbox = utils.im_transforms.ApplyExpand(img)
            roi = _project_im_rois(roi, expand_bbox)
        # vis(img, roi, show_name='expand')

        #-------------------------------------------------------------
        if cfg.TRAIN.USE_SAMPLE:
            sampled_bboxes = utils.im_transforms.GenerateBatchSamples(
                roi, img.shape)
            if len(sampled_bboxes) > 0:
                rand_idx = npr.randint(len(sampled_bboxes))
                sampled_bbox = sampled_bboxes[rand_idx]

                img = utils.im_transforms.Crop(img, sampled_bbox)
                roi = _project_im_rois(roi, sampled_bbox)
        # vis(img, roi, show_name='sample')

        # crop_bbox is define as RoIs with form (x1,y1,x2,y2)
        # if cfg.TRAIN.USE_CROP:
            # img, crop_bbox = utils.im_transforms.ApplyCrop(img)
            # roi = _project_im_rois(roi, crop_bbox)
        # vis(img, roi, show_name='crop')

        #-------------------------------------------------------------
        target_size = cfg.TRAIN.SCALES[random_scale_inds[i_im]]
        img, img_scale = utils.im_transforms.prep_im_for_blob(
            img, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)

        processed_ims.append(img)

        roi = _project_im_rois(
            roi, [0, 0, np.finfo, np.finfo], img_scale=img_scale)
        # vis(img, roi, show_name='prep', pixel_means=cfg.PIXEL_MEANS[0])

        #-------------------------------------------------------------
        # Check RoI
        datasets.ds_utils.validate_boxes(
            roi, width=img.shape[1], height=img.shape[0])

        # 归一化
        # roi_n = utils.im_transforms.normalize_img_roi(roi, img.shape)

        if cfg.CONTEXT:
            roi_inner, roi_outer = get_inner_outer_roi(roi, cfg.CONTEXT_RATIO)

        batch_ind = i_im * np.ones((roi.shape[0], 1))
        roi_blob_this_image = np.hstack((batch_ind, roi))
        roi_blob = np.vstack((roi_blob, roi_blob_this_image))

        if cfg.CONTEXT:
            roi_context_blob_this_image = np.hstack((batch_ind, roi_outer,
                                                     roi))
            roi_context_blob = np.vstack((roi_context_blob,
                                          roi_context_blob_this_image))

            roi_frame_blob_this_image = np.hstack((batch_ind, roi, roi_inner))
            roi_frame_blob = np.vstack((roi_frame_blob,
                                        roi_frame_blob_this_image))

        roi_score_blob = np.vstack((roi_score_blob, roi_score))

        im_roi_num = np.ones((1))
        im_roi_num[0] = roi.shape[0]
        roi_num_blob = np.vstack((roi_num_blob, im_roi_num))

    # Create a blob to hold the input images
    im_blob = im_list_to_blob(processed_ims)
    blobs = {'data': im_blob}

    blobs['roi'] = roi_blob
    if cfg.CONTEXT:
        blobs['roi_context'] = roi_context_blob
        blobs['roi_frame'] = roi_frame_blob

    # n * 1 to n
    blobs['roi_score'] = np.add(
        np.reshape(roi_score_blob, [roi_score_blob.shape[0]]), 1)

    roi_num_blob = np.reshape(roi_num_blob, [roi_num_blob.shape[0]])
    blobs['roi_num'] = roi_num_blob

    blobs['label'] = label_blob

    return blobs


def _project_im_rois(roi, crop_bbox, img_scale=[1, 1]):
    """Project image RoIs into the rescaled training image."""
    roi[:, 0] = np.minimum(np.maximum(roi[:, 0], crop_bbox[0]), crop_bbox[2])
    roi[:, 1] = np.minimum(np.maximum(roi[:, 1], crop_bbox[1]), crop_bbox[3])
    roi[:, 2] = np.maximum(np.minimum(roi[:, 2], crop_bbox[2]), crop_bbox[0])
    roi[:, 3] = np.maximum(np.minimum(roi[:, 3], crop_bbox[3]), crop_bbox[1])
    crop = np.tile(crop_bbox[:2], [roi.shape[0], 2])
    # roi = (roi - crop) * im_scale_factor
    roi = (roi - crop)

    roi[:, 0] = roi[:, 0] * img_scale[1]
    roi[:, 1] = roi[:, 1] * img_scale[0]
    roi[:, 2] = roi[:, 2] * img_scale[1]
    roi[:, 3] = roi[:, 3] * img_scale[0]

    # TODO(YH): 为什么大部分RoI会被caffe抛弃
    return roi


def get_inner_outer_roi(im_roi, ratio):
    assert ratio > 1, 'ratio should be lager than one in get_inner_outer_roi'
    roi = im_roi.astype(np.float32, copy=True)
    # x1 y1 x2 y2
    roi_w = roi[:, 2] - roi[:, 0]
    roi_h = roi[:, 3] - roi[:, 1]

    roi_inner_w = roi_w / ratio
    roi_inner_h = roi_h / ratio

    roi_outer_w = roi_w * ratio
    roi_outer_h = roi_h * ratio

    inner_residual_w = roi_w - roi_inner_w
    inner_residual_h = roi_h - roi_inner_h

    outer_residual_w = roi_outer_w - roi_w
    outer_residual_h = roi_outer_h - roi_h

    roi_inner = np.copy(roi)
    roi_outer = np.copy(roi)

    # print roi_inner.dtype, roi_inner.shape
    # print inner_residual_w.dtype, inner_residual_w.shape
    # print (inner_residual_w / 2).dtype, (inner_residual_w / 2).shape

    roi_inner[:, 0] += inner_residual_w / 2
    roi_inner[:, 1] += inner_residual_h / 2
    roi_inner[:, 2] -= inner_residual_w / 2
    roi_inner[:, 3] -= inner_residual_h / 2

    roi_outer[:, 0] -= outer_residual_w / 2
    roi_outer[:, 1] -= outer_residual_h / 2
    roi_outer[:, 2] += outer_residual_w / 2
    roi_outer[:, 3] += outer_residual_h / 2

    return roi_inner, roi_outer


def vis(img,
        rois,
        channel_swap=(0, 1, 2),
        pixel_means=np.zeros((1, 3)),
        show_name='image',
        normalized=False):
    im = img.copy()
    print show_name, ' mean: ', np.mean(im)
    num_roi_vis = 100
    # channel_swap = (0, 2, 3, 1)

    im = im.transpose(channel_swap)

    im += pixel_means.astype(im.dtype)
    im = im.astype(np.uint8).copy()

    height = im.shape[0]
    width = im.shape[1]

    num_img_roi = 0
    for j in range(rois.shape[0]):
        roi = rois[j, :]
        roi = np.squeeze(roi)
        num_img_roi += 1
        if num_img_roi > num_roi_vis:
            break
        if normalized:
            x1 = int(roi[0] * width)
            y1 = int(roi[1] * height)
            x2 = int(roi[2] * width)
            y2 = int(roi[3] * height)
        else:
            x1 = int(roi[0])
            y1 = int(roi[1])
            x2 = int(roi[2])
            y2 = int(roi[3])

        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imshow(show_name, im)
    cv2.waitKey(0)


def vis_minibatch(ims_blob,
                  rois_blob,
                  channel_swap=(0, 1, 2, 3),
                  pixel_means=np.zeros((1, 1, 3))):
    num_roi_vis = 100
    # channel_swap = (0, 2, 3, 1)

    ims = ims_blob.copy()
    ims = ims.transpose(channel_swap)
    ims += pixel_means

    for i in range(ims.shape[0]):
        im = ims[i]
        print 'wsl image mean: ', np.mean(ims)
        im = im.astype(np.uint8).copy()

        num_img_roi = 0
        for j in range(rois_blob.shape[0]):
            roi = rois_blob[j, :]
            roi = np.squeeze(roi)
            if roi[0] != i:
                continue
            num_img_roi += 1
            if num_img_roi > num_roi_vis:
                break
            x1 = int(roi[1])
            y1 = int(roi[2])
            x2 = int(roi[3])
            y2 = int(roi[4])
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.imshow('wsl image ' + str(i), im)
    cv2.waitKey(0)
