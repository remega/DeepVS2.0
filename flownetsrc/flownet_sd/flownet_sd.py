from ..net import Net, Mode
from ..utils import LeakyReLU, average_endpoint_error, pad, antipad
# from ..downsample import downsample
import math
import tensorflow as tf
slim = tf.contrib.slim


class FlowNetSD(Net):

    def __init__(self, mode=Mode.TRAIN, debug=False):
        super(FlowNetSD, self).__init__(mode=mode, debug=debug)

    def model(self, inputs, trainable=True):
        _, height, width, _ = inputs['input_a'].shape.as_list()
        with tf.variable_scope('FlowNetSD'):
            concat_inputs = tf.concat([inputs['input_a'], inputs['input_b']], axis=3)
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                # Only backprop this network if trainable
                                trainable=trainable,
                                # He (aka MSRA) weight initialization
                                weights_initializer=slim.variance_scaling_initializer(),
                                activation_fn=LeakyReLU,
                                # We will do our own padding to match the original Caffe code
                                padding='VALID'):

                weights_regularizer = slim.l2_regularizer(0.0004)
                with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                    conv0 = slim.conv2d(pad(concat_inputs), 64, 3, scope='conv0')
                    conv1 = slim.conv2d(pad(conv0), 64, 3, stride=2, scope='conv1')
                    conv1_1 = slim.conv2d(pad(conv1), 128, 3, scope='conv1_1')
                    conv2 = slim.conv2d(pad(conv1_1), 128, 3, stride=2, scope='conv2')
                    conv2_1 = slim.conv2d(pad(conv2), 128, 3, scope='conv2_1')
                    conv3 = slim.conv2d(pad(conv2_1), 256, 3, stride=2, scope='conv3')
                    conv3_1 = slim.conv2d(pad(conv3), 256, 3, scope='conv3_1')
                    conv4 = slim.conv2d(pad(conv3_1), 512, 3, stride=2, scope='conv4')
                    conv4_1 = slim.conv2d(pad(conv4), 512, 3, scope='conv4_1')
                    conv5 = slim.conv2d(pad(conv4_1), 512, 3, stride=2, scope='conv5')
                    conv5_1 = slim.conv2d(pad(conv5), 512, 3, scope='conv5_1')
                    # conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
                    # conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')

        return conv3_1, conv4_1, conv5_1#, conv6_1

