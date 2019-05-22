import os
import mxnet as mx
import logging
import time
from common import get_network
gpu = mx.gpu()
num_batches = 1000


def bench(model_name, batch_size):
    dtype='float32'
    net, data_shape = get_network(model_name, batch_size)
    mod = mx.mod.Module(symbol=net, context=gpu)
    mod.bind(for_training=False, inputs_need_grad=False, data_shapes=data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    # get data
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=gpu) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up

    for i in range(dry_run + num_batches):
        if i == dry_run:
            t = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
    mx.nd.waitall()
    return (time.time() - t) * 1000. / num_batches


if __name__ == '__main__':
    for batch in [1, 16]:
        for name in ['resnet-50', 'vgg-19', 'resnext-50', 'inception_v3', 'drn-c-26','dcn-resnet-101']:
            print('{} (batch={}): {} ms'.format(name, batch, bench(name, batch)))
