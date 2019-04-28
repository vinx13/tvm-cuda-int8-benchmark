from common import get_network
import os
import mxnet as mx


onnx_model_dir = 'onnx_model'

def prepare_onnx_model(model):
    path = os.path.join(onnx_model_dir, '{}.onnx'.format(model))
    if os.path.exists(path):
        return
    sym, data_shape = get_network(model, 1)
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0))
    mod.bind(for_training=False, inputs_need_grad=False, data_shapes=data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    arg_param, aux_param = mod.get_params()
    arg_param.update(aux_param)
    mx.contrib.onnx.mx2onnx.export_model.export_model(sym, params=arg_param, input_shape=[shape[1] for shape in data_shape], onnx_file_path=path)


def main():
    model_list = ['resnet-50', 'vgg-19', 'resnext-50', 'inception_v3', 'drn-c-26']

    for model in model_list:
        prepare_onnx_model(model)

    for batch in [1, 16]:
        for model in model_list:
            cmd = './TensorRT/bench {}.onnx {}'.format(os.path.join(onnx_model_dir, model), batch)
            os.system(cmd)

if __name__ == '__main__':
    main()
