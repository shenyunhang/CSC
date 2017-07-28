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
    roi_blob = np.zeros((0, 8), dtype=np.float32)
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

        # 处理标签
        gt_classes = roidb[i_im]['gt_classes']

        #-------------------------------------------------------------
        if cfg.TRAIN.USE_DISTORTION:
            img = utils.im_transforms.ApplyDistort(img)
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
            roi, gt_classes = _transform_img_roi(
                roi,
                gt_classes,
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
                roi, gt_classes = _transform_img_roi(
                    roi,
                    gt_classes,
                    do_crop=True,
                    crop_bbox=sampled_bbox,
                    img_shape=img.shape)
        # vis(img, roi, show_name='sample')

            #-------------------------------------------------------------
        target_size = cfg.TRAIN.SCALES[random_scale_inds[i_im]]
        img, img_scale = prep_im_for_blob(img, cfg.PIXEL_MEANS, target_size,
                                          cfg.TRAIN.MAX_SIZE)

        processed_ims.append(img)

        roi, gt_classes = _transform_img_roi(
            roi,
            gt_classes,
            do_resize=True,
            img_scale=img_scale,
            img_shape=img.shape)
        # vis(img, roi, show_name='prep', pixel_means=cfg.PIXEL_MEANS[0])

        #-------------------------------------------------------------
        # Check RoI
        datasets.ds_utils.validate_boxes(
            roi, width=img.shape[1], height=img.shape[0])

        # 归一化
        roi = _normalize_img_roi(roi, img.shape)

        #-------------------------------------------------------------
        instance_id = np.zeros_like(gt_classes)
        old_c = -1
        for c_i in range(gt_classes.shape[0]):
            c = gt_classes[c_i]
            if c == old_c:
                instance_id[c_i] = instance_id[c_i - 1] + 1
            else:
                old_c = c

        gt_classes = gt_classes.reshape((roi.shape[0], 1))
        instance_id = instance_id.reshape((roi.shape[0], 1))

        # TODO(YH): 目前全部设置不难
        difficult = np.zeros_like(gt_classes)

        batch_ind = i_im * np.ones((roi.shape[0], 1))
        roi_blob_this_image = np.hstack((batch_ind, gt_classes, instance_id,
                                         roi, difficult))
        roi_blob = np.vstack((roi_blob, roi_blob_this_image))

    # Create a blob to hold the input images
    im_blob = im_list_to_blob(processed_ims)
    blobs = {'data': im_blob}

    roi_blob = np.expand_dims(roi_blob, axis=0)
    roi_blob = np.expand_dims(roi_blob, axis=0)
    blobs['label'] = roi_blob

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

        height = im.shape[0]
        width = im.shape[1]

        num_img_roi = 0
        for j in range(rois_blob.shape[2]):
            roi = rois_blob[:, :, j, :]
            roi = np.squeeze(roi)
            if roi[0] != i:
                continue
            num_img_roi += 1
            if num_img_roi > num_roi_vis:
                break
            x1 = int(roi[3] * width)
            y1 = int(roi[4] * height)
            x2 = int(roi[5] * width)
            y2 = int(roi[6] * height)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.imshow('ssd image ' + str(i), im)
    cv2.waitKey(0)
