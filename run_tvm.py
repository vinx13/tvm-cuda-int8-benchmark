import numpy as np
import tvm
import tvm.autotvm as autotvm
import tvm.relay as relay
import tvm.relay.testing
import tvm.autotvm
from tvm.contrib import graph_runtime
from common import get_network
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--log_file', type=str, default='logs/history_best_1080.log')
args = parser.parse_args()

ctx = tvm.gpu(0)
target = tvm.target.cuda()

def bench(name, batch):
    sym, data_shape = get_network(name, batch)
    data_shape = data_shape[0][1]
    sym, _ = relay.frontend.from_mxnet(sym, {'data': data_shape})
    sym, params = tvm.relay.testing.create_workload(sym)
    with relay.quantize.qconfig(skip_k_conv=0, round_for_shift=True):
        sym = relay.quantize.quantize(sym, params)

    with relay.build_module.build_config(opt_level=3):
        graph, lib, params = relay.build(sym, 'cuda', 'llvm', params=params)

    m = graph_runtime.create(graph, lib, ctx)
    x = np.random.uniform(size=data_shape)
    data_tvm = tvm.nd.array(x.astype('float32'))
    m.set_input("data", data_tvm)
    m.set_input(**{k:tvm.nd.array(v, ctx) for k, v in params.items()})
    m.run()
    e = m.module.time_evaluator("run", ctx, number=2000, repeat=3)
    t = e(data_tvm).results
    t = np.array(t) * 1000

    print('{} (batch={}): {} ms'.format(name, batch, t.mean()))


def main():
    with tvm.target.cuda():
        with autotvm.apply_history_best(args.log_file):
            for batch in [1, 16]:
                for name in ['vgg-19', 'resnet-50', 'resnext-50', 'inception_v3', 'drn-c-26', 'dcn-resnet-101']:
                    bench(name, batch)


if __name__ == '__main__':
    main()
