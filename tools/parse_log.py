#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import argparse
import pprint
import re
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_log(logs, output_dir):
    path_to_trainlog = logs['train']

    regex_iteration = re.compile('Iteration (\d+)')
    regex_train_output = re.compile(
        'Train net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_test_output = re.compile(
        'Test net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_learning_rate = re.compile(
        'lr = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')

    # class prediction #roi #class predition pos neg #roi #left % class
    label = ['#class', 'prediction', '#roi', '#class',
             'prediction', 'pos', 'neg', '#roi', '#left', '%', 'class']
    regex_class_info_title = re.compile(
        '#class\tprediction\t#roi\t#class\tpredition\t\tpos\t\tneg\t\t#roi\t#left\t%\t\tclass')
    regex_class_info = re.compile(
        '(\d+)\t(\d*\.?\d+)\t(\d+)\t(\d+)\t(\d*\.?\d+)\t(\d*\.?\d+)\t(\d*\.?\d+)\t(\d+)\t(\d+)\t(\d*\.?\d+)\t(\w+)')
    regex_loss_pos = re.compile(
        '.+cross_entropy_loss_layer.+ loss_pos .+ AVE loss: (\d*\.?\d+)')
    regex_loss_neg = re.compile(
        '.+cross_entropy_loss_layer.+ loss_neg .+ AVE loss: (\d*\.?\d+)')

    regex_ap = re.compile('AP for (\w+) = (\d*\.?\d+) .+')

    info = dict()

    #------------------------------------------------------------------
    # parse train log
    iteration = -1
    print 'path_to_trainlog: ', path_to_trainlog
    with open(path_to_trainlog) as f:
        for line in f:
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
            if iteration == -1:
                # Only start parsing for other stuff if we've found the first
                # iteration
                continue

            class_info_match = regex_class_info.search(line)
            if class_info_match:
                print iteration, label
                print iteration, class_info_match.groups()
                class_info_this = class_info_match.groups()
                if class_info_this[-1] not in info:
                    info[class_info_this[-1]] = []

                info[class_info_this[-1]].append(
                    [float(x) for x in class_info_this[:-1]] + [0.0, iteration])
            else:
                print iteration, class_info_match

            loss_pos_match = regex_loss_pos.search(line)
            if loss_pos_match:
                print iteration, loss_pos_match.groups()
                if 'loss_pos' not in info:
                    info['loss_pos'] = []
                info['loss_pos'].append([iteration] + [float(x)
                                                       for x in loss_pos_match.groups()])
            else:
                print iteration, loss_pos_match

            loss_neg_match = regex_loss_neg.search(line)
            if loss_neg_match:
                print iteration, loss_neg_match.groups()
                if 'loss_neg' not in info:
                    info['loss_neg'] = []
                info['loss_neg'].append([iteration] + [float(x)
                                                       for x in loss_neg_match.groups()])
            else:
                print iteration, loss_neg_match

    #------------------------------------------------------------------
    # parse test log
    path_to_testlogs = logs['test']
    print path_to_testlogs
    for key in path_to_testlogs.keys():
        path_to_testlog = path_to_testlogs[key]
        print path_to_testlog
        iter_start = (key - 1) * 78
        iter_end = key * 78

        with open(path_to_testlog) as f:
            for line in f:
                ap_match = regex_ap.search(line)
                if ap_match:
                    cls = ap_match.group(1)
                    ap = ap_match.group(2)
                    info[cls][iter_end / 10 - 1][10] = ap

    #------------------------------------------------------------------
    # plot chart
    # pprint.pprint(info)
    print 'plot chart'

    label = ['#class', 'prediction', '#roi', '#class_cpg',
             'prediction_cpg', 'pos', 'neg', '#roi_cpg', '#left_cpg', '%', 'ap', 'iteration']
    for key in info.keys():
        if key in ['loss_pos', 'loss_neg']:
            continue
        data = np.array(info[key], dtype=np.float32)
        plt.clf()
        plt.figure(figsize=(15, 15))

        plt.subplot(4, 1, 1)
        plt.plot(data[:, -1], data[:, 0], label=label[0])
        plt.plot(data[:, -1], data[:, 3], label=label[3])
        plt.legend(loc='best')

        plt.subplot(4, 1, 2)
        plt.plot(data[:, -1], data[:, 1], label=label[1])
        plt.plot(data[:, -1], data[:, 4], label=label[4])
        plt.plot(data[:, -1], data[:, 5], label=label[5])
        plt.plot(data[:, -1], data[:, 6], label=label[6])
        plt.legend(loc='best')

        plt.subplot(4, 1, 3)
        plt.plot(data[:, -1], data[:, 2], label=label[2])
        plt.plot(data[:, -1], data[:, 7], label=label[7])
        plt.plot(data[:, -1], data[:, 8], label=label[8])
        plt.plot(data[:, -1], data[:, 9] * 1000, label=label[9])
        plt.legend(loc='best')

        x = data[:, -1]
        y = data[:, 10]
        x = x[np.nonzero(y)]
        y = y[np.nonzero(y)]
        plt.subplot(4, 1, 4)
        plt.plot(x, y, label=label[10])
        plt.legend(loc='best')

        plt.savefig(os.path.join(output_dir, key + '.png'))

    for key in info.keys():
        if key not in ['loss_pos', 'loss_neg']:
            continue

        data = np.array(info[key])
        plt.clf()
        plt.figure(figsize=(15, 15))
        plt.plot(data[:, 0], data[:, 1])

        plt.savefig(os.path.join(output_dir, key + '.png'))


def traversal(rootdir, framework, model):
    logs = dict()
    # regex_train = re.compile('train.+\.log')
    regex_train = re.compile('{}.sh_{}.+\.log'.format(framework, model))
    # regex_train = re.compile('^((?!test).)*\.log')
    regex_test = re.compile(
        '{}_test.sh_{}_{}_iter_(\d+).+\.log'.format(framework, model, model))

    # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所�  文件名字
    for parent, dirnames, filenames in os.walk(rootdir):
        for dirname in dirnames:  # 输出文件夹信息
            print "parent is: " + parent
            print "dirname is: " + dirname

        for filename in filenames:  # 输出文件信息
            if filename.find('.log') == -1:
                continue
            print "parent is: " + parent
            print "filename is: " + filename
            # 输出文件路径信息
            print "the full name of the file is: " + os.path.join(parent, filename)

            train_match = regex_train.search(filename)
            test_match = regex_test.search(filename)
            if train_match:
                print train_match.groups()
                logs['train'] = os.path.join(parent, filename)

            if test_match:
                iteration = int(test_match.groups()[0])
                if 'test' not in logs:
                    logs['test'] = dict()
                logs['test'][iteration] = os.path.join(parent, filename)

    pprint.pprint(logs)
    return logs


def parse_args():
    description = ('Parse a Caffe training log '
                   'containing training and testing information')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('log_dir',
                        help='Directory in which stored log files')

    parser.add_argument('output_dir',
                        help='Directory in which to place output files')

    parser.add_argument('framework',
                        help='framework name')
    parser.add_argument('model',
                        help='model name')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info (e.g., output filenames)')

    parser.add_argument('--delimiter',
                        default=',',
                        help=('Column delimiter in output files '
                              '(default: \'%(default)s\')'))

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    pprint.pprint(args)
    logs = traversal(args.log_dir, args.framework, args.model)
    parse_log(logs, args.output_dir)

if __name__ == '__main__':
    main()
