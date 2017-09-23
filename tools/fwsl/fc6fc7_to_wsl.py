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

import _init_paths
import caffe
import sys
print sys.argv

caffe_root = 'caffe-fwsl/'

if len(sys.argv) != 5:
    print 'usage:'
    print __file__, 'in.prototext in.caffemodel out.prototext out.caffemodel'
    exit(0)

in_prototext = sys.argv[1]
in_caffemodel = sys.argv[2]
out_prototext = sys.argv[3]
out_caffemodel = sys.argv[4]

# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net(in_prototext, in_caffemodel, caffe.TEST)
params = ['fc6', 'fc7', 'fc8c', 'fc8d']
# fc_params = {name: (weights, biases)}
fc_params = {
    pr: (net.params[pr][0].data, net.params[pr][1].data)
    for pr in params
}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(
        fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

# Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net(out_prototext, in_caffemodel, caffe.TEST)
params_full_conv = ['fc6_wsl', 'fc7_wsl', 'fc8c_wsl', 'fc8d_wsl']
# conv_params = {name: (weights, biases)}
conv_params = {
    pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data)
    for pr in params_full_conv
}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(
        conv, conv_params[conv][0].shape, conv_params[conv][1].shape)
# exit(0)

for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][
        0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

net_full_conv.save(out_caffemodel)
exit(0)

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# load input and configure preprocessing
im = caffe.io.load_image(caffe_root + '/examples/images/cat.jpg')
transformer = caffe.io.Transformer({
    'data':
    net_full_conv.blobs['data'].data.shape
})
transformer.set_mean(
    'data',
    np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(
        1).mean(1))
# transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale('data', 255.0)
# make classification map by forward and print prediction indices at each
# location
out = net_full_conv.forward_all(data=np.asarray(
    [transformer.preprocess('data', im)]))
print out['prob'][0].argmax(axis=0)
print out['prob'][0].shape
# show net input and confidence map (probability of the top prediction at
# each location)
plt.subplot(1, 2, 1)
plt.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
plt.subplot(1, 2, 2)
plt.imshow(out['prob'][0, 281])

plt.show()
