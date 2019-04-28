# Adapted from https://github.com/fyu/drn/blob/5a7228ee86ccc0d56d050308adaf25d5a6b9ef7f/drn.py

import math
import mxnet as mx


def conv3x3(data, out_planes, stride=1, padding=1, dilation=1):
    return mx.sym.Convolution(data, num_filter=out_planes, kernel=(3, 3), stride=(stride, stride),
                     pad=(padding, padding), no_bias=True, dilate=(dilation, dilation))


def BasicBlock(data, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
    expansion = 1
    conv1 = conv3x3(data, planes, stride=stride, padding=dilation[0], dilation=dilation[0])
    bn1 = mx.sym.BatchNorm(conv1)
    relu1 = mx.sym.Activation(data=bn1, act_type='relu')

    conv2 = conv3x3(relu1, planes, stride=1, padding=dilation[1], dilation=dilation[1])
    bn2 = mx.sym.BatchNorm(data=conv2)

    if downsample is not None:
        data = downsample(data)

    if residual:
        bn2 = bn2 + data

    relu2 = mx.sym.Activation(data=bn2, act_type='relu')

    return relu2


def Bottleneck(data, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
    expansion = 4
    conv1 = mx.sym.Convolution(data, num_filter=planes, kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn1 = mx.sym.BatchNorm(conv1)
    relu1 = mx.sym.Activation(data=bn1, act_type='relu')

    conv2 = mx.sym.Convolution(relu1, num_filter=planes, kernel=(3, 3), stride=(stride, stride), pad=(dilation[1], dilation[1]), no_bias=True, dilate=(dilation[1], dilation[1]))
    bn2 = mx.sym.BatchNorm(conv2)
    relu2 = mx.sym.Activation(data=bn2, act_type='relu')

    conv3 = mx.sym.Convolution(relu2, num_filter=planes * 4, kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn3 = mx.sym.BatchNorm(conv3)

    if downsample is not None:
        data = downsample(data)

    bn3 = bn3 + data
    relu3 = mx.sym.Activation(data=bn3, act_type='relu')

    return relu3


def DRN(block, layers, num_classes=1000,
        channels=(16, 32, 64, 128, 256, 512, 512, 512),
        out_map=False, out_middle=False, pool_size=28, arch='D'):
    inplanes = channels[0]
    out_map = out_map
    out_dim = channels[-1]
    out_middle = out_middle
    arch = arch

    def _make_layer(data, block, inplanes, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        expansion = 1 if block == BasicBlock else 4
        if stride != 1 or inplanes != planes * expansion:
            downsample = lambda data: mx.sym.BatchNorm(
                    mx.sym.Convolution(data, num_filter=planes * expansion, kernel=(1, 1), stride=(stride, stride), no_bias=True))

        layers = list()

        data = block(data, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (dilation // 2 if new_level else dilation, dilation),
            residual=residual)

        for i in range(1, blocks):
            data = block(data, planes, residual=residual,
                                dilation=(dilation, dilation))

        return data

    def _make_conv_layers(data, inplanes, channels, convs, stride=1, dilation=1):
        for i in range(convs):
            data = mx.sym.Convolution(num_filter=channels, kernel=(3, 3),
                          stride=(stride, stride) if i == 0 else (1, 1),
                          pad=(dilation, dilation), no_bias=True, dilate=(dilation, dilation))
            data = mx.sym.BatchNorm(data)
            data = mx.sym.Activation(data, act_type='relu')
        return data

    y = []

    data = mx.sym.Variable('data')
    conv1 = mx.sym.Convolution(data, num_filter=channels[0], kernel=(7, 7), stride=(1, 1), pad=(3, 3), no_bias=True)
    bn1 = mx.sym.BatchNorm(conv1)
    relu1 = mx.sym.Activation(bn1, act_type='relu')

    if arch == 'C':
        layer1 = _make_layer(
            relu1, BasicBlock, channels[0], channels[0], layers[0], stride=1)
        layer2 = _make_layer(
            layer1, BasicBlock, channels[0], channels[1], layers[1], stride=2)
    elif arch == 'D':
        layer1 = _make_conv_layers(
            relu1, channels[0], channels[0], layers[0], stride=1)
        layer2 = _make_conv_layers(
            layer1, channels[0], channels[1], layers[1], stride=2)

    layer3 = _make_layer(layer2, block, channels[1], channels[2], layers[2], stride=2)
    layer4 = _make_layer(layer3, block, channels[2], channels[3], layers[3], stride=2)
    layer5 = _make_layer(layer4, block, channels[3], channels[4], layers[4], dilation=2, new_level=False)

    y.append(layer1)
    y.append(layer2)
    y.append(layer3)
    y.append(layer4)
    y.append(layer5)

    if layers[5]:
        layer6 = _make_layer(layer5, block, channels[4], channels[5], layers[5], dilation=4, new_level=False)
        y.append(layer6)

    if arch == 'C':
        if layers[6]:
            layer7 = _make_layer(layer6, BasicBlock, channels[5], channels[6], layers[6], dilation=2,
                             new_level=False, residual=False)
            y.append(layer7)
        if layers[7]:
            layer8 = _make_layer(layer7, BasicBlock, channels[6], channels[7], layers[7], dilation=1,
                             new_level=False, residual=False)
            y.append(layer8)
    elif arch == 'D':
        if layers[6]:
            layer7 = _make_conv_layers(layer6, channels[5], channels[6], layers[6], dilation=2)
            y.append(layer7)
        if layers[7]:
            layer8 = _make_conv_layers(layer7, channels[6], channels[7], layers[7], dilation=1)
            y.append(layer8)

    out = y[-1]

    if out_map:
        out = mx.sym.Convolution(out, num_filter=num_classes, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=False)
    else:
        out = mx.sym.Pooling(out, kernel=(pool_size, pool_size), stride=(pool_size, pool_size), pool_type='avg')
        out = mx.sym.Convolution(out, num_filter=num_classes, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=False)
        out = mx.sym.flatten(out)

    if out_middle:
        return mx.sym.Group([out] + y)
    else:
        return out



def drn_c_26(**kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
    return model


def drn_c_42(**kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    return model


def drn_c_58(**kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    return model


def drn_d_22(**kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)
    return model


def drn_d_24(**kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', **kwargs)
    return model


def drn_d_38(**kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    return model


def drn_d_40(**kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    return model


def drn_d_54(**kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    return model


def drn_d_56(**kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    return model


def drn_d_105(**kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
    return model


def drn_d_107(**kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 2, 2], arch='D', **kwargs)
    return model
