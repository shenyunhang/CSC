# -*- coding: utf-8 -*-

import math
import numpy as np
import numpy.random as npr
import cv2
# from matplotlib.colors import rgb_to_hsv
# from matplotlib.colors import hsv_to_rgb
from configure import cfg

import utils.blob
# from caffe.io import resize_image


def GenerateBatchSamples(roi, img_shape):
    sampled_bboxes = []
    for i in range(len(cfg.TRAIN.batch_sampler)):
        sampled_bboxes_this = GenerateSamples(roi, cfg.TRAIN.batch_sampler[i],
                                              img_shape)
        sampled_bboxes.extend(sampled_bboxes_this)
    return sampled_bboxes


def GenerateSamples(roi, batch_sampler, img_shape):
    found = 0
    sampled_bboxes = []
    for i in range(batch_sampler.max_trials):
        if found > batch_sampler.max_sample:
            return sampled_bboxes
        # Generate sampled_bbox in the normalized space [0, 1].
        sampled_bbox = SampleBBox(batch_sampler.sampler, img_shape)

        if SatisfySampleConstraint(sampled_bbox, roi,
                                   batch_sampler.sample_constraint):
            found = found + 1
            sampled_bboxes.append(sampled_bbox)
    return sampled_bboxes


def SampleBBox(sampler, img_shape):
    # Get random scale.
    assert sampler.max_scale >= sampler.min_scale
    assert sampler.min_scale > 0.0
    assert sampler.max_scale <= 1.0
    scale = npr.uniform(sampler.min_scale, sampler.max_scale)

    # Get random aspect ratio.
    assert sampler.max_aspect_ratio >= sampler.min_aspect_ratio
    assert sampler.min_aspect_ratio > 0.0
    assert sampler.max_aspect_ratio < 10000
    aspect_ratio = npr.uniform(sampler.min_aspect_ratio,
                               sampler.max_aspect_ratio)

    aspect_ratio = max(aspect_ratio, 1.0 * math.pow(scale, 2.0))
    aspect_ratio = min(aspect_ratio, 1.0 / math.pow(scale, 2.0))

    # Figure out bbox dimension.
    bbox_width = scale * math.sqrt(aspect_ratio)
    bbox_height = scale / math.sqrt(aspect_ratio)

    # Figure out top left coordinates.
    h_off = npr.uniform(0.0, 1.0 - bbox_height)
    w_off = npr.uniform(0.0, 1.0 - bbox_width)

    #---------------------------------------
    bbox_height = bbox_height * img_shape[0]
    bbox_width = bbox_width * img_shape[1]
    h_off = h_off * img_shape[0]
    w_off = w_off * img_shape[1]

    assert bbox_width > 0
    assert bbox_height > 0

    sampled_bbox = np.array(
        [w_off, h_off, w_off + bbox_width, h_off + bbox_height],
        dtype=np.uint16)

    sampled_bbox[0] = min(max(sampled_bbox[0], 0), img_shape[1] - 1)
    sampled_bbox[1] = min(max(sampled_bbox[1], 0), img_shape[0] - 1)
    sampled_bbox[2] = min(
        max(sampled_bbox[2], sampled_bbox[0]), img_shape[1] - 1)
    sampled_bbox[3] = min(
        max(sampled_bbox[3], sampled_bbox[1]), img_shape[0] - 1)

    assert sampled_bbox[0] <= sampled_bbox[2]
    assert sampled_bbox[1] <= sampled_bbox[3]

    return sampled_bbox


def SatisfySampleConstraint(sampled_bbox, roi, sample_constraint):
    # Check constraints.
    found = False
    roi_num = roi.shape[0]
    for i in range(roi_num):
        this_roi = roi[i, :]
        jaccard_overlap = JaccardOverlap(sampled_bbox, this_roi)
        if jaccard_overlap < sample_constraint.min_jaccard_overlap:
            continue
        if jaccard_overlap > sample_constraint.max_jaccard_overlap:
            continue
        return True
    return False


def JaccardOverlap(bbox1, bbox2):
    intersect_bbox = IntersectBBox(bbox1, bbox2)
    intersect_width = intersect_bbox[2] - intersect_bbox[0] + 1
    intersect_height = intersect_bbox[3] - intersect_bbox[1] + 1

    if intersect_width > 0 and intersect_height > 0:
        intersect_size = intersect_width * intersect_height
        bbox1_size = BBoxSize(bbox1)
        bbox2_size = BBoxSize(bbox2)
        return 1.0 * intersect_size / (
            bbox1_size + bbox2_size - intersect_size)
    else:
        return 0.0


def IntersectBBox(bbox1, bbox2):
    if bbox2[0] > bbox1[2] or bbox2[2] < bbox1[0] or bbox2[1] > bbox1[3] or bbox2[3] < bbox1[1]:
        # Return [0, 0, 0, 0] if there is no intersection.
        # intersect_bbox=[0.0,0.0,0.0,0.0]
        intersect_bbox = [-1.0, -1.0, -1.0, -1.0]
    else:
        intersect_bbox = [
            max(bbox1[0], bbox2[0]),
            max(bbox1[1], bbox2[1]),
            min(bbox1[2], bbox2[2]),
            min(bbox1[3], bbox2[3])
        ]
    return intersect_bbox


def BBoxSize(bbox):
    if (bbox[2] < bbox[0] or bbox[3] < bbox[1]):
        # If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        return 0.0
    else:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return (width + 1) * (height + 1)


def Crop(img, crop_bbox):
    img_shape = img.shape
    # x1 = 1.0 * crop_bbox[0] * img_shape[1]
    # y1 = 1.0 * crop_bbox[1] * img_shape[0]
    # x2 = 1.0 * crop_bbox[2] * img_shape[1]
    # y2 = 1.0 * crop_bbox[3] * img_shape[0]

    x1 = crop_bbox[0]
    y1 = crop_bbox[1]
    x2 = crop_bbox[2]
    y2 = crop_bbox[3]

    assert x1 >= 0, x1
    assert y1 >= 0, y1
    assert x2 <= img_shape[1], '{} vs {}'.format(x2, img_shape[1])
    assert y2 <= img_shape[0], '{} vs {}'.format(y2, img_shape[0])

    crop_img = img[y1:y2 + 1, x1:x2 + 1, :]

    return crop_img


def MeetEmitConstraint(src_bbox, bbox):
    x_center = 1.0 * (bbox[0] + bbox[2]) / 2
    y_center = 1.0 * (bbox[1] + bbox[3]) / 2
    if x_center >= src_bbox[0] and x_center <= src_bbox[2] and y_center >= src_bbox[1] and y_center <= src_bbox[3]:
        return True
    else:
        return False


def ApplyCrop(img):
    if cfg.TRAIN.CROP <= 0:
        img_height = img.shape[0]
        img_width = img.shape[1]
        return img, np.array(
            (0, 0, img_width - 1, img_height - 1), dtype=np.uint16)

    img_shape = np.array(img.shape)
    crop_dims = img_shape[:2] * cfg.TRAIN.CROP
    # crop_dims = img_shape[:2] * 0.9

    r0 = npr.random()
    r1 = npr.random()
    s = img_shape[:2] - crop_dims
    s[0] *= r0
    s[1] *= r1
    # im_crop = np.array([s[0],
    # s[1],
    # s[0] + crop_dims[0] - 1,
    # s[1] + crop_dims[1] - 1],
    # dtype=np.uint16)

    crop_bbox = np.array(
        [s[1], s[0], s[1] + crop_dims[1] - 1, s[0] + crop_dims[0] - 1],
        dtype=np.uint16)
    crop_img = img[crop_bbox[1]:crop_bbox[3] + 1, crop_bbox[0]:
                   crop_bbox[2] + 1, :]

    return crop_img, crop_bbox


def ApplyExpand(img):
    img_shape = img.shape
    prob = npr.random()
    if prob > cfg.TRAIN.expand_prob:
        return img, np.array(
            (0, 0, img_shape[1], img_shape[0]), dtype=np.uint16)

    if abs(cfg.TRAIN.max_expand_ratio - 1.) < 1e-2:
        return img, np.array(
            (0, 0, img_shape[1], img_shape[0]), dtype=np.uint16)

    expand_ratio = npr.uniform(1, cfg.TRAIN.max_expand_ratio)
    expand_img, expand_bbox = ExpandImage(img, expand_ratio)
    return expand_img, expand_bbox


def ExpandImage(img, expand_ratio):
    img_height = img.shape[0]
    img_width = img.shape[1]
    img_channels = img.shape[2]

    # Get the bbox dimension.
    height = int(img_height * expand_ratio)
    width = int(img_width * expand_ratio)

    h_off = npr.uniform(0, height - img_height)
    w_off = npr.uniform(0, width - img_width)
    h_off = int(h_off)
    w_off = int(w_off)

    expand_bbox = []
    # expand_bbox.append(1.0 * (-w_off) / img_width)
    # expand_bbox.append(1.0 * (-h_off) / img_height)
    # expand_bbox.append(1.0 * (width - w_off) / img_width)
    # expand_bbox.append(1.0 * (height - h_off) / img_height)
    expand_bbox.append(-w_off)
    expand_bbox.append(-h_off)
    expand_bbox.append(width - w_off - 1)
    expand_bbox.append(height - h_off - 1)
    expand_bbox = np.array(expand_bbox)

    expand_img = np.tile(cfg.PIXEL_MEANS, (height, width, 1)).astype(img.dtype)

    expand_img[h_off:h_off + img_height, w_off:w_off + img_width, :] = img

    return expand_img, expand_bbox


def ApplyDistort_old(in_img):
    hsv = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)
    s0 = npr.random() * (cfg.TRAIN.SATURATION - 1) + 1
    s1 = npr.random() * (cfg.TRAIN.EXPOSURE - 1) + 1
    # s0 = npr.random() * (1.5 - 1) + 1
    # s1 = npr.random() * (1.5 - 1) + 1
    s0 = s0 if npr.random() > 0.5 else 1.0 / s0
    s1 = s1 if npr.random() > 0.5 else 1.0 / s1
    hsv = np.array(hsv, dtype=np.float32)
    hsv[:, :, 1] = np.minimum(s0 * hsv[:, :, 1], 255)
    hsv[:, :, 2] = np.minimum(s1 * hsv[:, :, 2], 255)
    hsv = np.array(hsv, dtype=np.uint8)
    out_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out_img


def ApplyDistort(in_img):
    prob = npr.random()
    if prob > 0.5:
        # Do random brightness distortion.
        out_img = RandomBrightness(in_img, cfg.TRAIN.brightness_prob,
                                   cfg.TRAIN.brightness_delta)
        # cv2.imshow('0 RandomBrightness',out_img.astype(np.uint8))

        # Do random contrast distortion.
        out_img = RandomContrast(out_img, cfg.TRAIN.contrast_prob,
                                 cfg.TRAIN.contrast_lower,
                                 cfg.TRAIN.contrast_upper)
        # cv2.imshow('1 RandomContrast',out_img.astype(np.uint8))

        # Do random saturation distortion.
        out_img = RandomSaturation(out_img, cfg.TRAIN.saturation_prob,
                                   cfg.TRAIN.saturation_lower,
                                   cfg.TRAIN.saturation_upper)
        # cv2.imshow('2 RandomSaturation',out_img.astype(np.uint8))

        # Do random exposure distortion.
        out_img = RandomExposure(out_img, cfg.TRAIN.exposure_prob,
                                 cfg.TRAIN.exposure_lower,
                                 cfg.TRAIN.exposure_upper)
        # cv2.imshow('3 RandomExposure',out_img.astype(np.uint8))

        # Do random hue distortion.
        out_img = RandomHue(out_img, cfg.TRAIN.hue_prob, cfg.TRAIN.hue_delta)
        # cv2.imshow('4 RandomHue',out_img.astype(np.uint8))

        # Do random reordering of the channels.
        out_img = RandomOrderChannels(out_img, cfg.TRAIN.random_order_prob)
        # cv2.imshow('5 RandomOrderChannels',out_img.astype(np.uint8))
    else:
        # Do random brightness distortion.
        out_img = RandomBrightness(in_img, cfg.TRAIN.brightness_prob,
                                   cfg.TRAIN.brightness_delta)
        # cv2.imshow('0 RandomBrightness',out_img.astype(np.uint8))

        # Do random saturation distortion.
        out_img = RandomSaturation(out_img, cfg.TRAIN.saturation_prob,
                                   cfg.TRAIN.saturation_lower,
                                   cfg.TRAIN.saturation_upper)
        # cv2.imshow('1 RandomSaturation',out_img.astype(np.uint8))

        # Do random exposure distortion.
        out_img = RandomExposure(out_img, cfg.TRAIN.exposure_prob,
                                 cfg.TRAIN.exposure_lower,
                                 cfg.TRAIN.exposure_upper)
        # cv2.imshow('2 RandomExposure',out_img.astype(np.uint8))

        # Do random hue distortion.
        out_img = RandomHue(out_img, cfg.TRAIN.hue_prob, cfg.TRAIN.hue_delta)
        # cv2.imshow('3 RandomHue',out_img.astype(np.uint8))

        # Do random contrast distortion.
        out_img = RandomContrast(out_img, cfg.TRAIN.contrast_prob,
                                 cfg.TRAIN.contrast_lower,
                                 cfg.TRAIN.contrast_upper)
        # cv2.imshow('4 RandomContrast',out_img.astype(np.uint8))

        # Do random reordering of the channels.
        out_img = RandomOrderChannels(out_img, cfg.TRAIN.random_order_prob)
        # cv2.imshow('5 RandomOrderChannels',out_img.astype(np.uint8))

    return out_img


def convertTo(in_img, alpha, beta):
    out_img = in_img.astype(np.float32)
    out_img = out_img * alpha + beta
    out_img = np.clip(out_img, 0, 255)
    out_img = out_img.astype(in_img.dtype)
    return out_img


# def bgr_to_hsv(bgr):
    # b, g, r = cv2.split(bgr)
    # rgb = cv2.merge((r, g, b))
    # hsv = rgb_to_hsv(rgb)
    # return hsv


# def hsv_to_bgr(hsv):
    # rgb = hsv_to_rgb(hsv)
    # r, g, b = cv2.split(rgb)
    # bgr = cv2.merge((b, g, r))
    # return bgr


def RandomBrightness(in_img, brightness_prob, brightness_delta):
    prob = npr.random()
    if prob < brightness_prob:
        assert brightness_delta > 0, "brightness_delta must be non-negative."
        delta = npr.uniform(-brightness_delta, brightness_delta)
        out_img = AdjustBrightness(in_img, delta)
    else:
        out_img = in_img
    return out_img


def AdjustBrightness(in_img, delta):
    if abs(delta) > 0:
        # out_img = cv2.convertTo(in_img, 1, 1, delta)
        out_img = convertTo(in_img, 1, delta)
    else:
        out_img = in_img
    return out_img


def RandomContrast(in_img, contrast_prob, lower, upper):
    prob = npr.random()
    if prob < contrast_prob:
        assert upper >= lower, 'contrast upper must be >= lower.'
        assert lower >= 0, 'contrast lower must be non-negative.'
        delta = npr.uniform(lower, upper)
        out_img = AdjustContrast(in_img, delta)
    else:
        out_img = in_img
    return out_img


def AdjustContrast(in_img, delta):
    if abs(delta - 1.0) > 1e-3:
        # out_img = cv2.convertTo(in_img, -1, delta, 0)
        out_img = convertTo(in_img, delta, 0)
    else:
        out_img = in_img
    return out_img


def RandomExposure(in_img, exposure_prob, lower, upper):
    prob = npr.random()
    if prob < exposure_prob:
        assert upper >= lower, 'saturation upper must be >= lower.'
        assert lower >= 0, 'saturation lower must be non-negative.'
        delta = npr.uniform(lower, upper)
        out_img = AdjustExposure(in_img, delta)
    else:
        out_img = in_img
    return out_img


def AdjustExposure(in_img, delta):
    if abs(delta - 1.0) != 1e-3:
        # Convert to HSV colorspae.
        out_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)
        # out_img = bgr_to_hsv(in_img)

        # Split the image to 3 channels.
        h, s, v = cv2.split(out_img)

        # Adjust the exposure.
        # channels[2] = cv2.convertTo(channels[2], -1, delta, 0)
        v = convertTo(v, delta, 0)
        # out_img = cv2.merge((h, s, v))
        out_img[:, :, 2] = v

        # Back to BGR colorspace.
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2BGR)
        # out_img = hsv_to_bgr(out_img)
    else:
        out_img = in_img
    return out_img


def RandomSaturation(in_img, saturation_prob, lower, upper):
    prob = npr.random()
    if prob < saturation_prob:
        assert upper >= lower, 'saturation upper must be >= lower.'
        assert lower >= 0, 'saturation lower must be non-negative.'
        delta = npr.uniform(lower, upper)
        out_img = AdjustSaturation(in_img, delta)
    else:
        out_img = in_img
    return out_img


def AdjustSaturation(in_img, delta):
    if abs(delta - 1.0) != 1e-3:
        # Convert to HSV colorspae.
        out_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)
        # out_img = bgr_to_hsv(in_img)

        # Split the image to 3 channels.
        h, s, v = cv2.split(out_img)

        # Adjust the saturation.
        # channels[1] = cv2.convertTo(channels[1], -1, delta, 0)
        s = convertTo(s, delta, 0)
        # out_img = cv2.merge((h, s, v))
        out_img[:, :, 1] = s

        # Back to BGR colorspace.
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2BGR)
        # out_img = hsv_to_bgr(out_img)
    else:
        out_img = in_img
    return out_img


def RandomHue(in_img, hue_prob, hue_delta):
    prob = npr.random()
    if prob < hue_prob:
        assert hue_delta >= 0, 'hue_delta must be non-negative.'
        delta = npr.uniform(-hue_delta, hue_delta)
        out_img = AdjustHue(in_img, delta)
    else:
        out_img = in_img
    return out_img


def AdjustHue(in_img, delta):
    if abs(delta) > 0:
        # Convert to HSV colorspae.
        out_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)
        # out_img = bgr_to_hsv(in_img)

        # Split the image to 3 channels.
        h, s, v = cv2.split(out_img)

        # Adjust the hue.
        # channels[0] = cv2.convertTo(channels[0], -1, 1, delta)
        h = convertTo(h, 1, delta)
        # out_img = cv2.merge((h, s, v))
        out_img[:, :, 0] = h

        # Back to BGR colorspace.
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2BGR)
        # out_img = hsv_to_bgr(out_img)
    else:
        out_img = in_img
    return out_img


def RandomOrderChannels(in_img, random_order_prob):
    prob = npr.random()
    if prob < random_order_prob:
        # Split the image to 3 channels.
        channels = cv2.split(in_img)
        assert len(channels) == 3

        # Shuffle the channels.
        channels = npr.shuffle(channels)
        out_img = cv2.merge(channels)
    else:
        out_img = in_img
    return out_img


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

    # if len(cfg.TRAIN.INTERP_MODEL) > 0:
    # interp_order = np.random.randint(0, 6)

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
        # im = resize_image(
        # im, (target_size, target_size), interp_order=interp_order)
        im_scales = [im_scale_h, im_scale_w]
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
        # im = resize_image(
        # im, (im_shape[0] * im_scale, im_shape[1] * im_scale),
        # interp_order=interp_order)
        im_scales = [im_scale, im_scale]
    else:
        print 'Unknow resize mode.'
        exit()

    return im, im_scales


def get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape

    processed_ims = []
    im_scale_factors = []

    if cfg.RESIZE_MODE == 'WARP':
        for target_size in cfg.TEST.SCALES:
            im_scale_h = float(target_size) / float(im_shape[0])
            im_scale_w = float(target_size) / float(im_shape[1])
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_w,
                fy=im_scale_h,
                interpolation=cv2.INTER_LINEAR)
            im_scale = [im_scale_h, im_scale_w]

            im_scale_factors.append(im_scale)
            processed_ims.append(im)
    elif cfg.RESIZE_MODE == 'FIT_SMALLEST':
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(
                im_orig,
                None,
                None,
                fx=im_scale,
                fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
            im_scale = [im_scale, im_scale]

            im_scale_factors.append(im_scale)
            processed_ims.append(im)

    # Create a blob to hold the input images
    blob = utils.blob.im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def normalize_img_roi(img_roi, img_shape):
    roi_normalized = np.copy(img_roi)
    roi_normalized[:, 0] = roi_normalized[:, 0] / img_shape[1]
    roi_normalized[:, 1] = roi_normalized[:, 1] / img_shape[0]
    roi_normalized[:, 2] = roi_normalized[:, 2] / img_shape[1]
    roi_normalized[:, 3] = roi_normalized[:, 3] / img_shape[0]
    return roi_normalized
