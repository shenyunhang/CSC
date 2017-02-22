#!/usr/bin/env python
#-*-coding:utf-8-*-
'''
                   _ooOoo_
                  o8888888o
                  88" . "88
                  (| -_- |)
                  O\  =  /O
               ____/`---'\____
             .'  \\|     |//  `.
            /  \\|||  :  |||//  \
           /  _||||| -:- |||||-  \
           |   | \\\  -  /// |   |
           | \_|  ''\---/''  |   |
           \  .-\__  `-`  ___/-. /
         ___`. .'  /--.--\  `. . __
      ."" '<  `.___\_<|>_/___.'  >'"".
     | | :  `- \`.;`\ _ /`;.`/ - ` : | |
     \  \ `-.   \_ __\ /__ _/   .-` /  /
======`-.____`-.___\_____/___.-`____.-'======
                   `=---='
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         佛祖保佑       永无BUG
'''
# Make sure that caffe is on the python path:
# this file is expected to be in {caffe_root}/examples
caffe_root = 'caffe-wsl/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
print sys.argv

if len(sys.argv) != 5:
    print 'usage:'
    print __file__, 'in.prototext in.caffemodel out.prototext out.caffemodel'
    exit(0)

in_prototext = sys.argv[1]
in_caffemodel = sys.argv[2]
out_prototext = sys.argv[3]
out_caffemodel = sys.argv[4]


# Load the original network and extract the fully connected layers' parameters.
net_in = caffe.Net(in_prototext, in_caffemodel, caffe.TEST)
params = ['fc6', 'fc7', 'fc8']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net_in.params[pr][0].data, net_in.params[
                  pr][1].data) for pr in params}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

# Load the out network to transplant the parameters.
net_out = caffe.Net(out_prototext, in_caffemodel, caffe.TEST)
params_new = ['fc6_1', 'fc7_1', 'fc8_1']
# conv_params = {name: (weights, biases)}
fc_params_new = {pr: (net_out.params[pr][0].data, net_out.params[
    pr][1].data) for pr in params_new}

for fc in fc_params_new:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params_new[fc][0].shape, fc_params_new[fc][1].shape)
# exit(0)

for pr, pr_new in zip(params, params_new):
    fc_params_new[pr_new][0].flat = fc_params[
        pr][0].flat  # flat unrolls the arrays
    fc_params_new[pr_new][1][...] = fc_params[pr][1]

net_out.save(out_caffemodel)
# exit(0)
