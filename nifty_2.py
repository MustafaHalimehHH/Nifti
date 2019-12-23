# https://github.com/NifTK/NiftyNetModelZoo/blob/master/ultrasound_simulator_gan_model_zoo.md
from __future__ import print_function, unicode_literals, division, absolute_import
import os
import numpy
import keras
import tensorflow


class ImageDiscriminator():
    def __init(self, name='discriminator'):
        self.initializer = {
            'w': tensorflow.contrib.layers.variance_scaling_initializer(),
            'b': tensorflow.constant_initializer(0)
        }

    def layer_op(self, image, conditioning, is_training):
        conditioning = tensorflow.image.resize_images(conditioning, [160, 120])
        batch_size = image.get_shape().as_list()[0]
        w_init = tensorflow.random_normal_initializer(0.0, 0.02)
        b_init = tensorflow.random_normal_initializer(0.0, 0.02)
        ch = [32, 64, 128, 256, 512, 1024]

        def leaky_relu(x, alpha=0.2):
            with tensorflow.name_scope('leaky_relu'):
                return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

        def down(ch, x):
            with tensorflow.name_scope('downsample'):
                x_ch = x.shape.as_list()[-1]
                conv_kernel = tensorflow.get_variable('w', shape=(x.shape[1], x.shape[2], 3), initializer=w_init,
                                                      regularizer=None)
                c = tensorflow.nn.convolution(input=x, filter=conv_kernel, strides=2, name='conv')
                c = tensorflow.contrib.layers.batch_norm(c)
                c = leaky_relu(c)
                return c

        def convr(ch, x):
            conv_kernel = tensorflow.get_variable('w', shape=(x.shape[1], x.shape[2], 3), initializer=w_init,
                                                  regularizer=None)
            c = tensorflow.nn.convolution(input=x, filter=conv_kernel, strides=2, name='conv')
            return leaky_relu(tensorflow.contrib.layers.batch_norm(c))

        def conv(ch, x, s):
            conv_kernel = tensorflow.get_variable('w', shape=(x.shape[1], x.shape[2], 3), initializer=w_init,
                                                  regularizer=None)
            c = tensorflow.nn.convolution(input=x, filter=conv_kernel, strides=2, name='conv')
            return leaky_relu(tensorflow.contrib.layers.batch_norm(c) + s)

        def down_blocks(ch, x):
            with tensorflow.name_scope('down_resnet'):
                s = down(ch, x)
                r = convr(ch, s)
                return conv(ch, r, s)

        if not conditioning is None:
            image = tensorflow.concat([image, conditioning], axis=1)
        with tensorflow.name_scope('feature'):
            conv_kernel = tensorflow.get_variable('w', shape=(image.shape[1], image.shape[2], 5), initializer=w_init,
                                                  regularizer=None)
            d_h1s = tensorflow.nn.convolution(input=image, filters=conv_kernel, strides=2, name='conv')
            d_h1s = leaky_relu(d_h1s)
            d_h1r = convr(ch[0], d_h1s)
            d_h1 = conv(ch[0], d_h1r, d_h1s)
        d_h2 = down_blocks(ch[1], d_h1)
        d_h3 = down_blocks(ch[2], d_h2)
        d_h4 = down_blocks(ch[3], d_h3)
        d_h5 = down_blocks(ch[4], d_h4)
        d_h6 = down_blocks(ch[5], d_h5)
        with tensorflow.name_scope('fc'):
            d_hf = tensorflow.reshape(d_h6, [batch_size, -1])
            d_nf_o = numpy.prod(d_hf.get_shape().as_list()[1:])
            D_wo = tensorflow.get_varaible('D_Wo', shape=[d_nf_o, 1], initializer=w_init)
            D_bo = tensorflow.get_variable('D_bo', shape=[1], initializer=b_init)
            d_logit = tensorflow.matmul(d_hf, D_wo) + D_bo
        return d_logit



