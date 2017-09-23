import argparse
import os
import shutil
import subprocess
import sys

import _init_paths
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from xml.etree.ElementTree import parse, Element
import cv2
import numpy as np
import math
from easydict import EasyDict as edict
import utils.im_transforms


def random_sample(height, width):
    sampler = edict()
    sampler.max_scale = 1.0
    sampler.min_scale = 0.1
    sampler.max_aspect_ratio = 10.0
    sampler.min_aspect_ratio = 0.1
    box = utils.im_transforms.SampleBBox(sampler, [height, width])
    return box[0] + 1, box[1] + 1, box[2] + 1, box[3] + 1


def random_sift(xmin, ymin, xmax, ymax, height, width):
    h = 1.0 * (ymax - ymin + 1)
    w = 1.0 * (xmax - xmin + 1)
    yctr = ymin + h / 2.0
    xctr = xmin + w / 2.0

    if math.ceil(w / 2.0) < width - math.ceil(w / 2.0):
        xctr = np.random.randint(
            math.ceil(w / 2.0), width - math.ceil(w / 2.0))

    if math.ceil(h / 2.0) < height - math.ceil(h / 2.0):
        yctr = np.random.randint(
            math.ceil(h / 2.0), height - math.ceil(h / 2.0))
    print xctr, yctr

    xmin = xctr - w / 2.0
    ymin = yctr - h / 2.0
    xmax = xctr + w / 2.0
    ymax = yctr + h / 2.0

    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)

    xmin = max(1, xmin)
    ymin = max(1, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)

    assert xmin > 0
    assert ymin > 0
    assert ymin < ymax
    assert xmin < xmax
    assert ymax <= height
    assert xmax <= width

    return xmin, ymin, xmax, ymax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create AnnotatedDatum database")
    parser.add_argument(
        "root",
        help="The root directory which contains the images and annotations.")
    parser.add_argument(
        "listfile",
        help="The file which contains image paths and annotation info.")
    parser.add_argument(
        "outdir", help="The output directory which stores the database file.")

    args = parser.parse_args()
    root_dir = args.root
    list_file = args.listfile
    out_dir = args.outdir

    # check if root directory exists
    if not os.path.exists(root_dir):
        print "root directory: {} does not exist".format(root_dir)
        sys.exit()
    # add "/" to root directory if needed
    if root_dir[-1] != "/":
        root_dir += "/"
    # check if list file exists
    if not os.path.exists(list_file):
        print "list file: {} does not exist".format(list_file)
        sys.exit()
    # check list file format is correct
    with open(list_file, "r") as lf:
        for line in lf.readlines():
            img_file, anno = line.strip("\n").split(" ")
            if not os.path.exists(root_dir + img_file):
                print "image file: {} does not exist".format(
                    root_dir + img_file)
                sys.exit()
            if not os.path.exists(root_dir + anno):
                print "annofation file: {} does not exist".format(
                    root_dir + anno)
                sys.exit()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # make some noise
    noise_prob = 0.5

    num = 0
    with open(list_file, "r") as lf:
        for line in lf.readlines():
            img_file, anno = line.strip("\n").split(" ")
            print num, line

            img = cv2.imread(os.path.join(root_dir, img_file))
            img_height, img_width, _ = img.shape

            doc = parse(os.path.join(root_dir, anno))
            root = doc.getroot()
            height = int(root.find('size').find('height').text)
            width = int(root.find('size').find('width').text)
            print height, width
            assert height == img_height, 'height mismatch {} vs {}: {}'.format(
                height, img_height, line)
            assert width == img_width, 'width mismatch {} vs {}: {}'.format(
                width, img_width, line)
            for v1 in root:
                if v1.tag != 'object':
                    continue
                print v1.tag, v1.attrib
                for v2 in v1:
                    if v2.tag != 'bndbox':
                        continue
                    if np.random.random() > noise_prob:
                        continue
                    print v2.tag, v2.attrib
                    xmin = int(v2.find('xmin').text)
                    ymin = int(v2.find('ymin').text)
                    xmax = int(v2.find('xmax').text)
                    ymax = int(v2.find('ymax').text)
                    print xmin, ymin, xmax, ymax

                    # xmin, ymin, xmax, ymax = random_sift(
                    # xmin, ymin, xmax, ymax, height, width)
                    xmin, ymin, xmax, ymax = random_sample(height, width)

                    v2.find('xmin').text = str(xmin)
                    v2.find('ymin').text = str(ymin)
                    v2.find('xmax').text = str(xmax)
                    v2.find('ymax').text = str(ymax)

                    print xmin, ymin, xmax, ymax

            _, anno_name = os.path.split(anno)
            save_path = os.path.join(out_dir, anno_name)
            print save_path
            doc.write(save_path)

            num = num + 1

    head, list_name = os.path.split(list_file)
    list_name_noise = 'noise_{}'.format(list_name)
    list_file_noise = os.path.join(head, list_name_noise)
    ori_dir = os.path.abspath(root_dir)
    with open(list_file, "r") as lf, open(list_file_noise, 'w') as lfn:
        for line in lf.readlines():
            img_file, anno = line.strip("\n").split(" ")

            _, anno_name = os.path.split(anno)
            save_path = os.path.join(out_dir, anno_name)
            abs_path = os.path.abspath(save_path)
            rel_path = os.path.relpath(abs_path, ori_dir)

            lfn.write('{} {}\n'.format(img_file, rel_path))
