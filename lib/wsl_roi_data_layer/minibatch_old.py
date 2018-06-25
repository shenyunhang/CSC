# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as npr
import cv2
import os
from configure import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import datasets.ds_utils


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    # rois_per_image = cfg.TRAIN.ROIS_PER_IM

    # Get the input image blob, formatted for caffe
    # im_crops is define as RoIs with form (y1,x1,y2,x2)
    im_blob, im_scales, im_crops, im_shapes = _get_image_blob(
        roidb, random_scale_inds)

    # row col row col to x1 y1 x2 y2
    im_crops = np.array(im_crops, dtype=np.uint16)
    im_crops = im_crops[:, (1, 0, 3, 2)]

    blobs = {'data': im_blob}

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    rois_context_blob = np.zeros((0, 9), dtype=np.float32)
    rois_frame_blob = np.zeros((0, 9), dtype=np.float32)
    rois_scores_blob = np.zeros((0, 1), dtype=np.float32)
    rois_num_blob = np.zeros((0, 1), dtype=np.float32)
    labels_blob = np.zeros((0, num_classes), dtype=np.float32)
    opg_filter_blob = np.zeros((0, num_classes), dtype=np.float32)
    opg_io_blob = np.zeros((0, 1), dtype=np.float32)
    for i_im in xrange(num_images):
        # x1 y1 x2 y2
        im_rois = roidb[i_im]['boxes'].astype(np.float32)
        im_labels = roidb[i_im]['gt_classes']
        if cfg.USE_ROI_SCORE:
            im_roi_scores = roidb[i_im]['box_scores']

        im_crop = im_crops[i_im]

        # TODO(YH): CROP is conflict with OPG_CACHE and ROI_AU, thereforce caffe should check the validity of RoI
        # 删除超出CROP的RoI
        # drop = (im_rois[:, 0] >= im_crop[2]) | (im_rois[:, 1] >= im_crop[3]) | (
        # im_rois[:, 2] <= im_crop[0]) | (im_rois[:, 3] <= im_crop[1])
        # im_rois = im_rois[~drop]
        # if cfg.USE_ROI_SCORE:
        # im_roi_scores = im_roi_scores[~drop]

        # Check RoI
        datasets.ds_utils.validate_boxes(
            im_rois, width=im_shapes[i_im][1], height=im_shapes[i_im][0])

        rois_per_this_image = np.minimum(cfg.TRAIN.ROIS_PER_IM,
                                         im_rois.shape[0])
        im_rois = im_rois[:rois_per_this_image, :]
        if cfg.USE_ROI_SCORE:
            im_roi_scores = im_roi_scores[:rois_per_this_image]

        if cfg.TRAIN.OPG_CACHE:
            filter_blob_this = np.zeros(
                (rois_per_this_image, num_classes), dtype=np.float32)
            for target_size in cfg.TRAIN.SCALES:
                if target_size == cfg.TRAIN.SCALES[random_scale_inds[i_im]]:
                    continue
                filter_name = str(db_inds[i_im] * 10000 + target_size)
                # print filter_name
                filter_path = os.path.join(cfg.TRAIN.OPG_CACHE_PATH,
                                           filter_name)

                if os.path.exists(filter_path):
                    filter_this = cpg.cpg_utils.binaryfile_to_blobproto_to_array(
                        filter_path).astype(np.float32)
                    # filter_blob_this = np.logical_or(
                    # filter_blob_this,
                    # cpg.cpg_utils.binaryfile_to_blobproto_to_array(filter_path)).astype(np.float32)
                    # filter_blob_this = np.add(
                    # filter_blob_this,
                    # cpg.cpg_utils.binaryfile_to_blobproto_to_array(filter_path)).astype(np.float32)
                    filter_blob_this = np.maximum(filter_blob_this,
                                                  filter_this)
            io_blob_this = np.array(
                [
                    db_inds[i_im] * 10000 +
                    cfg.TRAIN.SCALES[random_scale_inds[i_im]]
                ],
                dtype=np.float32)

            opg_filter_blob = np.vstack((opg_filter_blob, filter_blob_this))
            opg_io_blob = np.vstack((opg_io_blob, io_blob_this))

        if cfg.TRAIN.ROI_AU:
            offset = 1.0 / im_scales[i_im] / cfg.SPATIAL_SCALE
            offset_step = cfg.TRAIN.ROI_AU_STEP

            if cfg.TRAIN.OPG_CACHE:
                filter_blob_this_sum = np.sum(filter_blob_this, 1)
                au_ind = filter_blob_this_sum == 0
            else:
                au_ind = np.ones(rois_per_this_image, dtype=np.bool)
            offsets = np.random.randint(
                2 * offset_step + 1, size=(np.sum(au_ind),
                                           4)).astype(np.float32)
            offsets -= offset_step
            offsets *= offset

            au_rois_o = im_rois[au_ind]
            au_rois_n = im_rois[~au_ind]
            au_rois = au_rois_o + offsets

            keep = datasets.ds_utils.filter_validate_boxes(
                au_rois, im_shapes[i_im][1], im_shapes[i_im][0])
            au_rois[~keep] = au_rois_o[~keep]

            ovrs = datasets.ds_utils.overlaps(au_rois, au_rois_n)
            thresholded = ovrs >= 0.5
            keep = np.sum(thresholded, 1) == 0
            au_rois[~keep] = au_rois_o[~keep]

            # im_rois = np.vstack((im_rois, au_rois))
            im_rois[au_ind] = au_rois

            # rois_per_this_image = np.minimum(cfg.ROIS_PER_IM, im_rois.shape[0])
            # im_rois = im_rois[:rois_per_this_image, :]
            # if cfg.USE_ROI_SCORE:
            # au_roi_scores = im_roi_scores[au_ind]
            # im_roi_scores = np.vstack((im_roi_scores, au_roi_scores))
            # im_roi_scores = im_roi_scores[:rois_per_this_image]

            # roidb[i_im]['boxes'] = im_rois

        if cfg.CONTEXT:
            im_inner_rois, im_outer_rois = get_inner_outer_rois(
                im_rois, cfg.CONTEXT_RATIO)

        # project
        rois = _project_im_rois(im_rois, im_scales[i_im], im_crop)
        if cfg.CONTEXT:
            rois_inner = _project_im_rois(im_inner_rois, im_scales[i_im],
                                          im_crop)
            rois_outer = _project_im_rois(im_outer_rois, im_scales[i_im],
                                          im_crop)

        batch_ind = i_im * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))
        if cfg.CONTEXT:
            rois_context_blob_this_image = np.hstack((batch_ind, rois_outer,
                                                      rois))
            rois_context_blob = np.vstack((rois_context_blob,
                                           rois_context_blob_this_image))

            rois_frame_blob_this_image = np.hstack((batch_ind, rois,
                                                    rois_inner))
            rois_frame_blob = np.vstack((rois_frame_blob,
                                         rois_frame_blob_this_image))

        if cfg.USE_ROI_SCORE:
            rois_scores_blob = np.vstack((rois_scores_blob, im_roi_scores))
        else:
            rois_scores_blob = np.vstack((rois_scores_blob, np.zeros(
                (rois_per_this_image, 1), dtype=np.float32)))

        # Add to labels
        if cfg.USE_BG:
            im_labels = np.hstack((im_labels, [1.0]))
        labels_blob = np.vstack((labels_blob, im_labels))

        im_roi_num = np.ones((1))
        im_roi_num[0] = rois.shape[0]
        rois_num_blob = np.vstack((rois_num_blob, im_roi_num))
    # For debug visualizations
    # _vis_minibatch(im_blob, rois_blob, labels_blob)

    blobs['rois'] = rois_blob
    if cfg.CONTEXT:
        blobs['roi_context'] = rois_context_blob
        blobs['roi_frame'] = rois_frame_blob

    if cfg.USE_ROI_SCORE:
        # n * 1 to n
        blobs['roi_scores'] = np.add(
            np.reshape(rois_scores_blob, [rois_scores_blob.shape[0]]), 1)
    else:
        blobs['roi_scores'] = np.ones((rois_blob.shape[0]), dtype=np.float32)

    blobs['roi_num'] = rois_num_blob

    blobs['labels'] = labels_blob
    if cfg.TRAIN.OPG_CACHE:
        blobs['opg_filter'] = opg_filter_blob
        blobs['opg_io'] = opg_io_blob

    # print "rois_blob: ", rois_blob
    # print "rois_context_blob: ", rois_context_blob
    # print "rois_frame_blob: ", rois_frame_blob
    # print "rois_scores_blob: ", rois_scores_blob
    # print "rois_num_blob: ", rois_num_blob
    # print "labels_blob: ", labels_blob

    if cfg.TRAIN.ROI_AU:
        return blobs, roidb
    return blobs


def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    im_crops = []
    im_shapes = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_shapes.append(im.shape)

        if cfg.TRAIN.USE_DISTORTION:
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            s0 = npr.random() * (cfg.TRAIN.SATURATION - 1) + 1
            s1 = npr.random() * (cfg.TRAIN.EXPOSURE - 1) + 1
            s0 = s0 if npr.random() > 0.5 else 1.0 / s0
            s1 = s1 if npr.random() > 0.5 else 1.0 / s1
            hsv = np.array(hsv, dtype=np.float)
            hsv[:, :, 1] = np.minimum(s0 * hsv[:, :, 1], 255)
            hsv[:, :, 2] = np.minimum(s1 * hsv[:, :, 2], 255)
            hsv = np.array(hsv, dtype=np.uint8)
            im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if cfg.TRAIN.USE_CROP:
            im_shape = np.array(im.shape)
            crop_dims = im_shape[:2] * cfg.TRAIN.CROP

            r0 = npr.random()
            r1 = npr.random()
            s = im_shape[:2] - crop_dims
            s[0] *= r0
            s[1] *= r1
            im_crop = np.array(
                [s[0], s[1], s[0] + crop_dims[0] - 1, s[1] + crop_dims[1] - 1],
                dtype=np.uint16)

            im = im[im_crop[0]:im_crop[2] + 1, im_crop[1]:im_crop[3] + 1, :]
        else:
            im_crop = np.array(
                [0, 0, im.shape[0] - 1, im.shape[1] - 1], dtype=np.uint16)

        if cfg.OPG_DEBUG:
            im_save = im

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)

        if cfg.OPG_DEBUG:
            im_save = cv2.resize(
                im_save,
                None,
                None,
                fx=im_scale,
                fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
            cv2.imwrite('tmp/' + str(cfg.TRAIN.PASS_IM) + '_.png', im_save)
            cfg.TRAIN.PASS_IM = cfg.TRAIN.PASS_IM + 1

        im_scales.append(im_scale)
        im_crops.append(im_crop)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales, im_crops, im_shapes


def get_inner_outer_rois(im_rois, ratio):
    assert ratio > 1, 'ratio should be lager than one in get_inner_outer_rois'
    rois = im_rois.astype(np.float32, copy=True)
    # x1 y1 x2 y2
    rois_w = rois[:, 2] - rois[:, 0]
    rois_h = rois[:, 3] - rois[:, 1]

    rois_inner_w = rois_w / ratio
    rois_inner_h = rois_h / ratio

    rois_outer_w = rois_w * ratio
    rois_outer_h = rois_h * ratio

    inner_residual_w = rois_w - rois_inner_w
    inner_residual_h = rois_h - rois_inner_h

    outer_residual_w = rois_outer_w - rois_w
    outer_residual_h = rois_outer_h - rois_h

    rois_inner = np.copy(rois)
    rois_outer = np.copy(rois)

    # print rois_inner.dtype, rois_inner.shape
    # print inner_residual_w.dtype, inner_residual_w.shape
    # print (inner_residual_w / 2).dtype, (inner_residual_w / 2).shape

    rois_inner[:, 0] += inner_residual_w / 2
    rois_inner[:, 1] += inner_residual_h / 2
    rois_inner[:, 2] -= inner_residual_w / 2
    rois_inner[:, 3] -= inner_residual_h / 2

    rois_outer[:, 0] -= outer_residual_w / 2
    rois_outer[:, 1] -= outer_residual_h / 2
    rois_outer[:, 2] += outer_residual_w / 2
    rois_outer[:, 3] += outer_residual_h / 2

    return rois_inner, rois_outer


def _project_im_rois(im_rois, im_scale_factor, im_crop):
    """Project image RoIs into the rescaled training image."""
    im_rois[:, 0] = np.minimum(
        np.maximum(im_rois[:, 0], im_crop[0]), im_crop[2])
    im_rois[:, 1] = np.minimum(
        np.maximum(im_rois[:, 1], im_crop[1]), im_crop[3])
    im_rois[:, 2] = np.maximum(
        np.minimum(im_rois[:, 2], im_crop[2]), im_crop[0])
    im_rois[:, 3] = np.maximum(
        np.minimum(im_rois[:, 3], im_crop[3]), im_crop[1])
    crop = np.tile(im_crop[:2], [im_rois.shape[0], 2])
    rois = (im_rois - crop) * im_scale_factor

    # For YAROIPooling Layer
    # rois = (im_rois - crop)
    # width = im_crop[2] - im_crop[0]
    # height = im_crop[3] - im_crop[1]
    # rois[:, 0] = rois[:, 0] / width
    # rois[:, 1] = rois[:, 1] / height
    # rois[:, 2] = rois[:, 2] / width
    # rois[:, 3] = rois[:, 3] / height

    return rois


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
