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
        img = img.astype(np.float32)
        if roidb[i_im]['flipped']:
            img = img[:, ::-1, :]

        # 处理ROI
        # x1 y1 x2 y2
        roi = roidb[i_im]['boxes'].astype(np.float32)

        # Check RoI
        datasets.ds_utils.validate_boxes(
            roi, width=img.shape[1], height=img.shape[0])

        # vis(img, roi, show_name='origin')

        if cfg.USE_ROI_SCORE:
            roi_score = roidb[i_im]['box_scores']
        else:
            roi_score = np.zeros((roi.shape[0], 1), dtype=np.float32)

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
        # crop_bbox is define as RoIs with form (x1,y1,x2,y2)
        # if cfg.TRAIN.USE_CROP:
        # img, crop_bbox = utils.im_transforms.ApplyCrop(img)
        # else:
        # crop_bbox = np.array(
        # [0, 0, img.shape[0] - 1, img.shape[1] - 1], dtype=np.uint16)

        #-------------------------------------------------------------
        # expand_bbox is define as RoIs with form (x1,y1,x2,y2)
        if cfg.TRAIN.USE_EXPAND:
            img, expand_bbox = utils.im_transforms.ApplyExpand(img)
            roi, roi_score = _transform_img_roi(
                roi,
                roi_score,
                do_crop=True,
                crop_bbox=expand_bbox,
                img_shape=img.shape)
        # vis(img, roi, show_name='expand')

        #-------------------------------------------------------------
        if cfg.TRAIN.USE_SAMPLE:
            sampled_bboxes = utils.im_transforms.GenerateBatchSamples(
                roi, img.shape)
            if len(sampled_bboxes) > 0:
                rand_idx = npr.randint(len(sampled_bboxes))
                sampled_bbox = sampled_bboxes[rand_idx]

                img = utils.im_transforms.Crop(img, sampled_bbox)
                roi, roi_score = _transform_img_roi(
                    roi,
                    roi_score,
                    do_crop=True,
                    crop_bbox=sampled_bbox,
                    img_shape=img.shape)
        # vis(img, roi, show_name='sample')

        if cfg.TRAIN.USE_CROP:
            img, crop_bbox = utils.im_transforms.ApplyCrop_old(img)
            roi, roi_score = _transform_img_roi(
                roi,
                roi_score,
                do_crop=True,
                crop_bbox=crop_bbox,
                img_shape=img.shape)
        # vis(img, roi, show_name='crop')

        #-------------------------------------------------------------
        target_size = cfg.TRAIN.SCALES[random_scale_inds[i_im]]
        img, img_scale = prep_im_for_blob(img, cfg.PIXEL_MEANS, target_size,
                                          cfg.TRAIN.MAX_SIZE)

        processed_ims.append(img)

        roi, roi_score = _transform_img_roi(
            roi,
            roi_score,
            do_resize=True,
            img_scale=img_scale,
            img_shape=img.shape)
        # vis(img, roi, show_name='prep', pixel_means=cfg.PIXEL_MEANS[0])

        #-------------------------------------------------------------
        # Check RoI
        datasets.ds_utils.validate_boxes(
            roi, width=img.shape[1], height=img.shape[0])

        # 归一化
        # roi = _normalize_img_roi(roi, img.shape)

        #-------------------------------------------------------------
        roi_per_this_image = np.minimum(cfg.TRAIN.ROIS_PER_IM, roi.shape[0])
        roi = roi[:roi_per_this_image, :]
        roi_score = roi_score[:roi_per_this_image]

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

    blobs['roi_num'] = roi_num_blob

    blobs['label'] = label_blob

    return blobs


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape

    #-------------------------------------------------------------
    interp_mode = cv2.INTER_LINEAR
    if len(cfg.TRAIN.INTERP_MODEL) > 0:
        idx = npr.randint(len(cfg.TRAIN.INTERP_MODEL))
        interp_name = cfg.TRAIN.INTERP_MODEL[idx]
        if interp_name == 'LINEAR':
            interp_mode = cv2.INTER_LINEAR
        elif interp_name == 'AREA':
            interp_mode = cv2.INTER_AREA
        elif interp_name == 'NEAREST':
            interp_mode = cv2.INTER_NEAREST
        elif interp_name == 'CUBIC':
            interp_mode = cv2.INTER_CUBIC
        elif interp_name == 'LANCZOS4':
            interp_mode = cv2.INTER_LANCZOS4
        else:
            print 'Unknow interp mode: ', interp_name
            exit(0)

    if cfg.RESIZE_MODE == 'WARP':
        im_scale_h = float(target_size) / float(im_shape[0])
        im_scale_w = float(target_size) / float(im_shape[1])
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_w,
            fy=im_scale_h,
            interpolation=interp_mode)
        im_scale = [im_scale_h, im_scale_w]
    elif cfg.RESIZE_MODE == 'FIT_SMALLEST':
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=interp_mode)
        im_scale = [im_scale, im_scale]
    else:
        print 'Unknow resize mode.'

    return im, im_scale


def _normalize_img_roi(img_roi, img_shape):
    roi_normalized = np.copy(img_roi)
    roi_normalized[:, 0] = roi_normalized[:, 0] / img_shape[1]
    roi_normalized[:, 1] = roi_normalized[:, 1] / img_shape[0]
    roi_normalized[:, 2] = roi_normalized[:, 2] / img_shape[1]
    roi_normalized[:, 3] = roi_normalized[:, 3] / img_shape[0]
    return roi_normalized


def _transform_img_roi(roi,
                       score_or_label,
                       do_crop=False,
                       crop_bbox=None,
                       do_resize=False,
                       img_scale=[1, 1],
                       img_shape=[np.finfo, np.finfo]):
    if do_resize:
        roi = _UpdateBBoxByResizePolicy(roi, img_scale)

    if do_crop:
        roi, score_or_label = _project_img_roi(roi, score_or_label, crop_bbox)

    roi[:, 0] = np.minimum(np.maximum(roi[:, 0], 0), img_shape[1] - 1)
    roi[:, 1] = np.minimum(np.maximum(roi[:, 1], 0), img_shape[0] - 1)
    roi[:, 2] = np.minimum(np.maximum(roi[:, 2], 0), img_shape[1] - 1)
    roi[:, 3] = np.minimum(np.maximum(roi[:, 3], 0), img_shape[0] - 1)

    return roi, score_or_label


def _project_img_roi(roi, score_or_label, src_bbox):
    num_roi = roi.shape[0]
    roi_ = []
    score_or_label_ = []
    for i in range(num_roi):
        roi_this = roi[i, :]
        score_or_label_this = score_or_label[i]
        if utils.im_transforms.MeetEmitConstraint(src_bbox, roi_this):
            roi_.append(roi_this)
            score_or_label_.append(score_or_label_this)
    roi = np.array(roi_, dtype=np.float32)
    score_or_label = np.array(score_or_label_, dtype=np.float32)

    # assert roi.shape[0]>0
    if roi.shape[0] == 0:
        return np.zeros(
            (0, 4), dtype=np.float32), np.zeros(
                (0), dtype=np.float32)

    roi[:, 0] = 1.0 * (roi[:, 0] - src_bbox[0])
    roi[:, 1] = 1.0 * (roi[:, 1] - src_bbox[1])
    roi[:, 2] = 1.0 * (roi[:, 2] - src_bbox[0])
    roi[:, 3] = 1.0 * (roi[:, 3] - src_bbox[1])

    return roi, score_or_label


def _UpdateBBoxByResizePolicy(roi, img_scale):
    assert img_scale[0] > 0
    assert img_scale[1] > 0
    # new_shape = [
    # 1.0 * img_shape[0] * img_scale[0], 1.0 * img_shape[1] * img_scale[1]
    # ]

    # roi[:, 0] = roi[:, 0] * img_shape[1]
    # roi[:, 1] = roi[:, 1] * img_shape[0]
    # roi[:, 2] = roi[:, 2] * img_shape[1]
    # roi[:, 3] = roi[:, 3] * img_shape[0]

    roi[:, 0] = roi[:, 0] * img_scale[1]
    roi[:, 1] = roi[:, 1] * img_scale[0]
    roi[:, 2] = roi[:, 2] * img_scale[1]
    roi[:, 3] = roi[:, 3] * img_scale[0]

    # roi[:, 0] = roi[:, 0] / new_shape[1]
    # roi[:, 1] = roi[:, 1] / new_shape[0]
    # roi[:, 2] = roi[:, 2] / new_shape[1]
    # roi[:, 3] = roi[:, 3] / new_shape[0]

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

    im += pixel_means
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
