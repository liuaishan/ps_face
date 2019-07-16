import tensorflow as tf
import torch
import torch.nn as nn

WEIGHTS_INIT_STDEV = .1

def lrelu(x, leak=0.2, name="lrelu"):
    return torch.max(x, leak * x)

# batch normalization
# 我猜momentumu相当于是1-tf里面的decay
# 以下两个tf都有scale=true，要乘上gamma，什么意思以及该怎么办？
# 以及name应该没有影响吧。。。
def batch_norm(x,batch_size,momentum=0.95,epsilon=1e-5,name="batch_norm"):
    return nn.BatchNorm2d(
        num_features=batch_size, momentum=1 - momentum, eps=epsilon, name=name)(x)

# layer normalization
def layer_norm(x,shape):
    return nn.LayerNorm(normalized_shape=shape)(x)


def linear(input_, output_size, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    matrix = nn.init.normal_(torch.empty([shape[1], output_size], dtype=torch.float32), stddev)
    bias = nn.init.constant_(torch.empty([output_size]), bias_start)
    if with_w:
        return torch.matmul(input_, matrix) + bias, matrix, bias
    else:
        return torch.matmul(input_, matrix) + bias

def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = nn.init.normal_(torch.empty([1]), mean=0, std=0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                            kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = nn.init.normal_(torch.empty([1]), mean=0, std=0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                      kernel_initializer=initializer)

# convolution layer
def _conv_layer(net, num_filters, filter_size, strides, relu=True, name="conv2d"):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    net = tf.nn.conv2d(weights_init, padding='SAME')
    net = nn.Conv2d(stride=strides)(net)
    net = _instance_norm(net)
    if relu:
        return tf.nn.relu(net)
    else:
        return lrelu(net)


# deconvolution layer
def _conv_tranpose_layer(net, num_filters, filter_size, strides, relu=True, name="tansconv"):
    with tf.variable_scope(name):
        weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True, name=name)

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1, strides, strides, 1]

        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = _instance_norm(net)

        if relu:
            return tf.nn.relu(net)
        else:
            return lrelu(net)


# residual layer
def _residual_block(net, filter_size=3, name="residual"):
    tmp = _conv_layer(net, 128, filter_size, 1, name=name)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False, name=name)


def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    # 只对height和weight两个轴做平均和方差
    mu = torch.mean(net, [1, 2], keepdim=True)
    sigma_sq = torch.std(net, [1, 2], keepdim=True)
    shift = torch.zeros(var_shape)
    scale = torch.ones(var_shape)
    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
    return scale * normalized + shift


# initial weight for current layer
def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]
    weights_init = torch.clamp(nn.init.normal_(torch.empty(weights_shape, dtype=torch.float32), std=WEIGHTS_INIT_STDEV),
                               min=-2*WEIGHTS_INIT_STDEV, max=2*WEIGHTS_INIT_STDEV)
    return weights_init
