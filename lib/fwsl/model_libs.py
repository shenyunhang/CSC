import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2


def RoIPoolingLayer(net, feature_name, roi_name, out_name, pooled_h, pooled_w,
                    spatial_scale):

    net[out_name] = L.ROIPooling(
        net[feature_name],
        net[roi_name],
        pooled_h=pooled_h,
        pooled_w=pooled_w,
        spatial_scale=spatial_scale)

def ya_RoIPoolingLayer(net, feature_name, roi_name, out_name, pooled_h, pooled_w):

    net[out_name] = L.YAROIPooling(
        net[feature_name],
        net[roi_name],
        pooled_h=pooled_h,
        pooled_w=pooled_w)


def FcReluDropLayer(net,
                    bottom,
                    fc_name,
                    relu_name,
                    drop_name,
                    num_output,
                    lr_mult=1,
                    has_filler=False):
    if has_filler:
        kwargs = {
            'param': [
                dict(lr_mult=lr_mult, decay_mult=1),
                dict(lr_mult=2 * lr_mult, decay_mult=0)
            ],
            'weight_filler':
            dict(type='xavier'),
            'bias_filler':
            dict(type='constant', value=0),
        }
    else:
        kwargs = {
            'param': [
                dict(lr_mult=lr_mult, decay_mult=1),
                dict(lr_mult=2 * lr_mult, decay_mult=0)
            ],
        }
    net[fc_name] = L.InnerProduct(net[bottom], num_output=num_output, **kwargs)
    if len(relu_name) > 0:
        net[relu_name] = L.ReLU(net[fc_name], in_place=True)
        if len(drop_name) > 0:
            net[drop_name] = L.Dropout(
                net[relu_name], in_place=True, dropout_ratio=0.5)
        else:
            return
    else:
        if len(drop_name) > 0:
            net[drop_name] = L.Dropout(
                net[fc_name], in_place=True, dropout_ratio=0.5)
        else:
            return


def CreateRoIDataLayer(source,
                       num_class,
                       batch_size=32,
                       backend=P.Data.LMDB,
                       train=True,
                       label_map_file='',
                       anno_type=None,
                       transform_param={},
                       batch_sampler=[{}],
                       max_roi_per_im=2048,
                       visualize=False):
    if train:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            'transform_param': transform_param,
        }
    else:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
            'transform_param': transform_param,
        }
    ntop = 6
    annotated_data_param = {
        'label_map_file': label_map_file,
        'batch_sampler': batch_sampler,
    }

    roi_data_param = {
        'num_class': num_class,
        'max_roi_per_im': max_roi_per_im,
        'visualize': visualize,
    }

    if anno_type is not None:
        annotated_data_param.update({'anno_type': anno_type})

    return L.RoIData(
        name="data",
        annotated_data_param=annotated_data_param,
        roi_data_param=roi_data_param,
        data_param=dict(batch_size=batch_size, backend=backend, source=source),
        ntop=ntop,
        **kwargs)


def ya_VGGNetBody(net,
                  from_layer,
                  need_fc=True,
                  fully_conv=False,
                  reduced=False,
                  dilated=False,
                  nopool=False,
                  dropout=True,
                  lr_mult=1,
                  freeze_all_layers=True,
                  freeze_layers=[],
                  dilate_pool4=False):
    if freeze_all_layers:
        kwargs = {
            'param':
            [dict(lr_mult=0, decay_mult=0),
             dict(lr_mult=0, decay_mult=0)],
        }
    else:
        kwargs = {
            'param': [
                dict(lr_mult=lr_mult, decay_mult=1),
                dict(lr_mult=2 * lr_mult, decay_mult=0)
            ],
        }
    # kwargs = {
    # 'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
    # 'weight_filler': dict(type='xavier'),
    # 'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(
        net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(
        net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(
            net.relu1_2,
            num_output=64,
            pad=1,
            kernel_size=3,
            stride=2,
            **kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(
            net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(
        net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(
        net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(
            net.relu2_2,
            num_output=128,
            pad=1,
            kernel_size=3,
            stride=2,
            **kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(
            net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(
        net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(
        net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(
        net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(
            net.relu3_3,
            num_output=256,
            pad=1,
            kernel_size=3,
            stride=2,
            **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(
            net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(
        net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(
        net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(
        net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(
            net.relu4_3,
            num_output=512,
            pad=1,
            kernel_size=3,
            stride=2,
            **kwargs)
    else:
        name = 'pool4'
        if dilate_pool4:
            net[name] = L.Pooling(
                net.relu4_3,
                pool=P.Pooling.MAX,
                kernel_size=3,
                stride=1,
                pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(
                net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1

    kernel_size = 3
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    if dilation == 1:
        net.conv5_1 = L.Convolution(
            net[name],
            num_output=512,
            pad=pad,
            kernel_size=kernel_size,
            # dilation=dilation,
            **kwargs)
        net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
        net.conv5_2 = L.Convolution(
            net.relu5_1,
            num_output=512,
            pad=pad,
            kernel_size=kernel_size,
            # dilation=dilation,
            **kwargs)
        net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
        net.conv5_3 = L.Convolution(
            net.relu5_2,
            num_output=512,
            pad=pad,
            kernel_size=kernel_size,
            # dilation=dilation,
            **kwargs)
        net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)
    else:
        net.conv5_1 = L.Convolution(
            net[name],
            num_output=512,
            pad=pad,
            kernel_size=kernel_size,
            dilation=dilation,
            **kwargs)
        net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
        net.conv5_2 = L.Convolution(
            net.relu5_1,
            num_output=512,
            pad=pad,
            kernel_size=kernel_size,
            dilation=dilation,
            **kwargs)
        net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
        net.conv5_3 = L.Convolution(
            net.relu5_2,
            num_output=512,
            pad=pad,
            kernel_size=kernel_size,
            dilation=dilation,
            **kwargs)
        net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(
                    net.relu5_3,
                    num_output=512,
                    pad=1,
                    kernel_size=3,
                    stride=1,
                    **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(
                    net.relu5_3,
                    pool=P.Pooling.MAX,
                    pad=1,
                    kernel_size=3,
                    stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(
                    net.relu5_3,
                    num_output=512,
                    pad=1,
                    kernel_size=3,
                    stride=2,
                    **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(
                    net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    dilation = dilation * 6
                    kernel_size = 3
                    num_output = 1024
                else:
                    dilation = dilation * 2
                    kernel_size = 7
                    num_output = 4096
            else:
                if reduced:
                    dilation = dilation * 3
                    kernel_size = 3
                    num_output = 1024
                else:
                    kernel_size = 7
                    num_output = 4096
            pad = int((kernel_size + (dilation - 1) *
                       (kernel_size - 1)) - 1) / 2
            net.fc6 = L.Convolution(
                net[name],
                num_output=num_output,
                pad=pad,
                kernel_size=kernel_size,
                dilation=dilation,
                **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(
                    net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(
                    net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(
                    net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(
                    net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(
                    net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(
                    net.relu7, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [
        dict(lr_mult=0, decay_mult=0),
        dict(lr_mult=0, decay_mult=0)
    ]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net
