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
        conv1_1 = c