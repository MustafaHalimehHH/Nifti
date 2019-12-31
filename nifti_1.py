# https://github.com/rogertrullo/tensorflow_medical_images_segmentation
from __future__ import division
import os
import time
import glob
import tensorflow
import numpy
import collections
import threading
import datetime
import queue


def conv_op(input_op, name, kw, kh, n_out, dw, dh, wd, padding='SAME', activation=True):
    n_in = input_op.get_shape()[-1].value
    shape = [kh, kw, n_in, n_out]
    with tensorflow.variable_scope(name):
        # kernel = _variable_with_weight_decay('w', shape, dw)
        receptive_field_size = numpy.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
        init_range = numpy.sqrt(6.0 / (fan_in + fan_out))
        initializer = tensorflow.random_uniform_initializer(-init_range, init_range)
        var = tensorflow.get_variable('w', shape=shape, dtype=tensorflow.float32, initializer=initializer)
        weight_decay = tensorflow.mul(tensorflow.nn.l2_loss(var), wd, name='weight_loss')
        tensorflow.add_to_collection('losses', weight_decay)
        kernel = var
        conv = tensorflow.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding=padding)
        bias_init_val = tensorflow.constant(0.0, shape=[n_out], dtype=tensorflow.float32)
        biases = tensorflow.get_variable(initializer=bias_init_val, trainable=True, name='b')
        z = tensorflow.nn.bias_add(conv, biases)
        if activation:
            z = tensorflow.nn.relu(z, name='Activation')
        return z


def conv_op_bn(input_op, name, kw, kh, n_out, dw, dh, wd, padding, train_phase):
    n_in = input_op.get_shape()[-1].value
    shape = [kh, kw, n_in, n_out]
    scope_bn = name + '_bn'
    with tensorflow.variable_scope(name):
        receptive_field_size = numpy.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
        init_range = numpy.sqrt(6.0 / (fan_in + fan_out))
        initializer = tensorflow.random_uniform_initializer(-init_range, init_range)
        var = tensorflow.get_variable('w', shape=shape, dtype=tensorflow.float32, initializer=initializer)
        weight_decay = tensorflow.mul(tensorflow.nn.l2_loss(var), wd, name='weigh_loss')
        tensorflow.add_to_collection('losses', weight_decay)
        kernel = var
        conv = tensorflow.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding=padding)
        bias_init_val = tensorflow.constant(0.0, shape=[n_out], dtype=tensorflow.float32)
        biases = tensorflow.get_variable(initializer=bias_init_val, trainable=True, name='b')
        out_conv = tensorflow.nn.bias_add(conv, biases)
        z = tensorflow.layers.batch_normalization(out_conv, scale=False, center=False)
        return z


def deconv_op(input_op, name, kw, kh, n_out, wd, batchsize, activation=True):
    n_in = input_op.get_shape()[-1].value
    shape = [kh, kw, n_out, n_in]
    hin = input_op.get_shape()[1].value
    win = input_op.get_shape()[2].value
    output_shape = [batchsize, 2 * hin, 2 * win, n_out]
    with tensorflow.variable_scope(name):
        receptive_field_size = numpy.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
        init_range = numpy.sqrt(6.0 / (fan_in + fan_out))
        initializer = tensorflow.random_uniform_initializer(-init_range, init_range)
        var = tensorflow.get_variable('w', shape=shape, dtype=tensorflow.float32, initializer=initializer)
        kernel = var
        deconv = tensorflow.nn.conv2d_transpose(input_op, kernel, output_shape, strides=[1, 2, 2, 1], padding='SAME')
        bias_init_val = tensorflow.constant(0.0, shape=[n_out], dtype=tensorflow.float32)
        biases = tensorflow.get_variable(initializer=bias_init_val, trainable=True, name='b')
        z = tensorflow.nn.bias_add(deconv, biases)
        if activation:
            z = tensorflow.nn.relu(z, name='Activation')
        return z


def fully_connected_op(input_op, name, n_out, wd, activation=True):
    im_shape = input_op.get_shape().as_list()
    n_inputs = int(numpy.prod(im_shape[1:]))
    shape = [n_inputs, n_out]
    with tensorflow.variable_scope(name):
        receptive_field_size = numpy.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
        init_range = numpy.sqrt(6.0 / (fan_in + fan_out))
        initializer = tensorflow.random_uniform_initializer(-init_range, init_range)
        w = tensorflow.get_variable('w', shape=shape, dtype=tensorflow.float32, initializer=initializer)
        bias_init_val = tensorflow.constant(0.0, shape=[n_out], dtype=tensorflow.float32)
        biases = tensorflow.get_variable(initializer=bias_init_val, trainable=True, name='b')
        if len(im_shape) > 2:
            x = tensorflow.reshape(input_op, [-1, n_inputs])
        else:
            x = input_op
        z = tensorflow.matmul(x, w) + biases
        if activation:
            z = tensorflow.nn.relu(z)
        return z


def loss_dice(logits, labels, num_classes, batch_size_tf):
    with tensorflow.name_scope('loss'):
        probs = tensorflow.nn.softmax(logits)
        y_onehot = tensorflow.one_hot(labels, num_classes, 1.0, 0.0, axis=3, dtype=tensorflow.float32)
        print('probs shape ', probs.get_shape())
        print('y_onehot shape ', y_onehot.get_shape())
        num = tensorflow.reduce_sum(tensorflow.mul(probs, y_onehot), [1, 2])
        den1 = tensorflow.reduce_sum(tensorflow.mul(probs, probs), [1, 2])
        den2 = tensorflow.reduce_sum(tensorflow.mul(y_onehot, y_onehot), [1, 2])

        dice = 2 * (num/ (den1 + den2))
        dice_total = -1 * tensorflow.reduce_sum(dice, [1, 0]) / tensorflow.to_float(batch_size_tf)
        loss = dice_total
        return loss


def loss_fcn(logits, labels, num_classes, batch_size_tf, weights=None):
    with tensorflow.name_scope('loss'):
        shape_labels = labels.get_shape().as_list()
        logits = tensorflow.reshape(logits, [batch_size_tf * shape_labels[1] * shape_labels[2], num_classes])
        shape_labels = labels.get_shape().as_list()
        labels = tensorflow.reshape(labels, [batch_size_tf * shape_labels[1] * shape_labels[2]])
        labels_onehot = tensorflow.one_hot(labels, num_classes)

        if weights is not None:
            labels_weights = tensorflow.transpose(tensorflow.matmul(labels_onehot, weights))
            cross_entropy = labels_weights * tensorflow.nn.softmax_cross_entropy_with_logits(logits, labels_onehot, name=None)
        else:
            cross_entropy = tensorflow.nn.softmax_cross_entropy_with_logits(logits, labels_onehot, name=None)
        print(cross_entropy.get_shape())
        cross_entropy_mean = tensorflow.reduce_sum(cross_entropy, name='xentropy_mean') / tensorflow.to_float(batch_size_tf)
    return cross_entropy_mean


class seg_GAN(object):
    def __init__(self, sess, batch_size=10, height=512, width=512, wd=5e-4, checkpoint_dir=None, path_patients_h5=None, learning_rate=2e-8, lr_step=30000, lam_dice=1, lam_fcn=1, lam_adv=1,adversarial=False):
        self.sess = sess
        self.adversarial = adversarial
        self.lam_dice = lam_dice
        self.lam_fcn = lam_fcn
        self.lam_adv = lam_adv
        self.lr_step = lr_step
        self.wd = wd
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.checkpoint_dir = checkpoint_dir
        self.data_queue = queue.Queue(100)
        self.path_patients_h5 = path_patients_h5
        self.build_model()

    def build_model(self):
        self.class_weights = tensorflow.transpose(tensorflow.constant([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=tensorflow.float32, name='classweights'))
        self.num_classes = 5
        self.inputCT = tensorflow.placeholder(tensorflow.float32, shape=[None, self.height, self.width, 1])
        self.CT_GT = tensorflow.placeholder(tensorflow.int32, shape=[None, self.height, self.width])
        batch_size_tf = tensorflow.shape(self.inputCT)[0]
        self.train_phase = tensorflow.placeholder(tensorflow.bool, name='phase_train')
        self.G, self.layer = self.generator(self.inputCT, batch_size_tf)
        print('G shape', self.G.get_shape)
        self.prediction = tensorflow.argmax(self.G, 3)
        t_vars = tensorflow.trainable_variables()

        if self.adversarial:
            self.probs_G = tensorflow.nn.softmax(self.G)
            self.GT_1hot = tensorflow.one_hot(self.CT_GT, self.num_classes, 1.0, 0.0, axis=3, dtype=tensorflow.float32)
            print('GT_1hot shape', self.GT_1hot.get_shape())
            print('prediction shape', self.prediction.get_shape)
            self.D, self.D_logits = self.discriminator(self.GT_1hot)
            self.D_, self.D_logits_ = self.discriminator(self.probs_G, reuse=True)
            self.d_loss_real = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tensorflow.ones_like(self.D)))
            self.d_loss_fake = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tensorflow.zeros_like(self.D_)))
            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.g_loss, self.diceterm, self.fcnter, self.bceterm = self.combined_loss_G(batch_size_tf)

            self.d_vars = [var for var in t_vars if 'd_' in var.name]
            self.d_optim = tensorflow.train.AdamOptimizer(self.learning_rate, beta1=5e-1).minimize(self.d_loss, var_list=self.d_vars)
        else:
            self.g_loss = self.diceterm, self.fcnterm = self.combined_loss_G(batch_size_tf)

        self.global_step = tensorflow.Variable(0, name='global_step', trainable=False)
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        print('learning rate', self.learning_rate)
        self.learning_rate_tensor = tensorflow.train.exponential_decay(self.learning_rate, self.global_step, self.lr_step, 1e-1, staircase=True)
        self.g_optim = tensorflow.train.MomentumOptimizer(self.learning_rate_tensor, 9e-1).minimize(self.g_loss, global_step=self.global_step)
        self.merged = tensorflow.merge_all_summaries()
        self.writer = tensorflow.summary.FileWriter('./summaries', self.sess.graph)
        self.saver = tensorflow.train.Saver(max_to_keep=50000)

    def generator(self, input_op, batch_size_tf):
        conv1_1 = conv_op(input_op, name='g_conv1_1', kh=7, kw=7, n_out=32, dh=1, dw=1, wd=self.wd)
        conv1_2 = conv_op(conv1_1, name='g_conv1_2', kh=7, kw=7, n_out=32, dh=1, dw=1, wd=self.wd)
        pool1 = tensorflow.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='g_pool1')
        conv2_1 = conv_op(pool1, name='g_conv2_1', kh=7, kw=7, n_out=64, dh=1, dw=1, wd=self.wd)
        conv2_2 = conv_op(conv2_1, name='g_conv2_2', kh=7, kw=7, n_out=64, dh=1, dw=1, wd=self.wd)
        pool2 = tensorflow.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='g_pool2')
        conv3_1 = conv_op(pool2, name='g_conv3_1', kh=7, kw=7, n_out=96, dh=1, dw=1, wd=self.wd)
        conv3_2 = conv_op(conv3_1, name='g_conv3_2', kh=7, kw=7, n_out=96, dh=1, dw=1, wd=self.wd)
        pool3 = tensorflow.nn.max_pool(conv3_2, ksize=[1, 2 ,2 ,1], strides=[1, 2, 2, 1], padding='SAME', name='g_pool3')
        conv4_1 = conv_op(pool3, name='g_conv4_1', kh=7, kw=7, n_out=128, dh=1, dw=1, wd=self.wd)
        conv4_2 = conv_op(conv4_1, name='g_conv4_2', kh=7, kw=7, n_out=128, dh=1, dw=1, wd=self.wd)
        deconv1 = deconv_op(conv4_2, name='g_deconv1', kh=4, kw=4, n_out=64, wd=self.wd, batchsize=batch_size_tf)
        concat1 = tensorflow.concat(3, [deconv1, conv3_2], name='g_concat1')
        deconv2 = deconv_op(concat1, name='g_deconv2', kh=4, kw=4, n_out=64, wd=self.wd, batchsize=batch_size_tf)
        concat2 = tensorflow.concat(3, [deconv2, conv2_2], name='g_concat2')
        deconv3 = deconv_op(concat2, name='g_deconv3', kh=4, kw=4, n_out=32, wd=self.wd, batchsize=batch_size_tf)
        concat3 = tensorflow.concat(3, [deconv3, conv1_2], name='g_concat3')
        upsocre = conv_op(concat3, name='g_upscore', kh=7, kw=7, n_out=self.num_classes, dh=1, dw=1, wd=self.wd, activation=False)
        return upsocre, upsocre

    def discriminator(self, inputCT, reuse=False):
        if reuse:
            tensorflow.get_variable_scope().reuse_variables()
        print('ct_shape', inputCT.get_shape())
        h0 = conv_op_bn(inputCT, name='d_conv_dis_1_a', kh=5, kw=5, n_out=32, dh=1, dw=1, wd=self.wd, padding='VALID', train_phase=self.train_phase)
        m0 = tensorflow.nn.max_pool(h0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool0')
        h1 = conv_op_bn(m0, name='d_conv2_dis_a', kh=5, kw=5, n_out=64, dh=1, dw=1, wd=self.wd, padding='VALID', train_phase=self.train_phase)
        m1 = tensorflow.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool1')
        h2 = conv_op_bn(m1, name='d_conv3_dis_a', kh=5, kw=5, n_out=128, dh=1, dw=1, wd=self.wd, padding='VALID', train_phase=self.train_phase)
        h3 = conv_op_bn(h2, name='d_conv4_dis_a', kh=5, kw=5, n_out=64, dh=1, dw=1, wd=self.wd, padding='VALID', train_phase=self.train_phase)
        fc1 = fully_connected_op(h3, name='d_fc1', n_out=64, wd=self.wd, activation=True)
        fc2 = fully_connected_op(fc1, name='d_fc2', n_out=32, wd=self.wd, activation=True)
        fc3 = fully_connected_op(fc2, name='d_fc3', n_out=1, wd=self.wd, activation=False)
        return tensorflow.nn.sigmoid(fc3), fc3

    def train(self, config):
        pass
