from model import resnet, resnext, vgg, inception_v3, drn, dcn


def get_network(name, batch_size, num_classes=1000):
    image_shape = (3, 224, 224)
    if name == 'dcn-resnet-101':
        net = dcn.get_symbol(is_train=False)
    elif 'resnet' in name:
        n_layer = int(name.split('-')[1])
        net = resnet.get_symbol(num_classes=num_classes, num_layers=n_layer, image_shape='{},{},{}'.format(*image_shape))
    elif 'resnext' in name:
        n_layer = int(name.split('-')[1])
        net = resnext.get_symbol(num_classes=num_classes, num_layers=n_layer, image_shape='{},{},{}'.format(*image_shape))
    elif 'vgg' in name:
        n_layer = int(name.split('-')[1])
        net = vgg.get_symbol(num_classes=num_classes, num_layers=n_layer, batch_norm='bn' in name)
    elif name == 'inception_v3':
        image_shape = (3, 299, 299)
        net = inception_v3.get_symbol(num_classes=num_classes)
    elif name == 'drn-c-26':
        net = drn.drn_c_26()
    else:
        raise NotImplementedError()

    data_shape = [('data', (batch_size,) + image_shape)]

    return net, data_shape
