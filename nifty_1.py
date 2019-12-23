from __future__ import print_function, absolute_import, division, unicode_literals
import os
import numpy
import keras
import tensorflow
import matplotlib.pyplot as plt


def get_ckpt_vars(ckpt_path, output_file):
    # file = open(output_file, 'w+')
    ck = tensorflow.train.load_checkpoint(ckpt_path)
    print('type(ck)', type(ck))
    print('ck', ck)
    # tensorflow.train.list_variables(ckpt_path)
    print('trainable_variables', tensorflow.train.trainable_variables())
    for v in tensorflow.compat.v1.trainable_variables():
        print('var:', str(v))
        tensor_name = v.name.split(':')[0]
        print('tensor_name', tensor_name)
        if ck.has_tensor(tensor_name):
            print('__has_tensor({})'.format(tensor_name))
    '''
    for v in tensorflow.train.list_variables(ckpt_path):
        print('var:', str(v))
        tensor_name = v.name.split(':')[0]
        print('tensor_name', tensor_name)
        if ck.has_tensor(tensor_name):
            print('__has_tensor({})'.format(tensor_name))
    '''
    # file.close()


# ckpt_path = 'D:\\Halimeh\\NiftyNet\\Models\\ultrasound_simulator_gan_weights.tar\\ultrasound_simulator_gan_weights\\models\\model.ckpt-1.data-00000-of-00001'
# ckpt_path = 'D:\\Halimeh\\NiftyNet\\Models\\ultrasound_simulator_gan_weights.tar\\ultrasound_simulator_gan_weights\\models\\model.ckpt'
ckpt_path = 'D:\\Halimeh\\NiftyNet\\Models\\ultrasound_simulator_gan_weights\\models\\model.ckpt-1'
get_ckpt_vars(ckpt_path, '')
