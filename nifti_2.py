# https://github.com/Issam28/Brain-tumor-segmentation/blob/master/extract_patches.py
import os
import random
import nibabel
import numpy
import glob
import keras
import json

keras.backend.set_image_data_format("channel_last")

class Pipeline(object):
    def __init__(self, list_train, normalize=True):
        self.scans_train = list_train
        self.train_im = self.read_scans(normalize)

    def read_scans(self, normalize):
        train_im = []
        for i in range(len(self.scans_train)):
            if i % 10 == 0:
                print('iteration: [{}]'.format(i))

            flair = glob.glob(self.scans_train[i] + '/*_flair.nii.gz')
            t2 = glob.glob(self.scans_train[i] + '/*_t2.nii.gz')
            gt = glob.glob(self.scans_train[i] + '/*_seg.nii.gz')
            t1 = glob.glob(self.scans_train[i] + '/*_t1.nii.gz')
            t1c = glob.glob(self.scans_train[i] + '/*_t1ce.nii.gz')

            t1s = [scan for scan in t1 if scan not in t1c]
            if (len(flair) + len(t2) + len(gt) + len(t1s) + len(t1c)) < 5:
                print('a problem with patient {}'.format(self.scans_train[i]))
                continue

            scans = [flair[0], t1s[0], t1c[0], t2[0], gt[0]]
            tmp = [nibabel.load(scans[k]).get_fdata() for k in range(len(scans))]
            z0 = 1
            y0 = 29
            x0 = 42
            z1 = 147
            y1 = 221
            x1 = 194
            tmp = numpy.array(tmp)
            tmp = tmp[:, z0: z1, y0: y1, x0: x1]
            if normalize:
                tmp = self.norm_slices(tmp)
            train_im.append(tmp)
            del tmp
        return numpy.array(train_im)

    def sample_patches_randomly(self, num_patches, d, h, w):
        patches, labels = [], []
        count = 0
        gt_im = numpy.swapaxes(self.train_im, 0, 1)[4]
        msk = numpy.swapaxes(self.train_im, 0, 1)[0]
        tmp_shape = gt_im.shape
        gt_im = gt_im.reshape(-1).astype(numpy.uint8)
        msk = msk.reshape(-1).astype(numpy.float32)
        indices = numpy.squeeze(numpy.argwhere((msk != -9.0) & (msk != 0.0)))
        del msk
        numpy.random.shuffle(indices)
        gt_im = gt_im.reshape(tmp_shape)

        i = 0
        pix = len(patches)
        while (count < numpy) and (pix > i):
            ind = indices[i]
            i += 1
            ind = numpy.unravel_index(ind, tmp_shape)
            patient_id = ind[0]
            slice_idx = ind[1]
            p = ind[2:]
            p_y = (p[0] - (h)/2, p[0] + (h)/2)
            p_x = (p[1] - (w)/2, p[1] + (w)/2)
            p_x = list(map(int, p_x))
            p_y = list(map(int, p_y))

            tmp = self.train_im[patient_id][0:4, slice_idx, p_y[0]: p_y[1], p_x[0]: p_x[1]]
            lbl = gt_im[patient_id, slice_idx, p_y[0]: p_y[1], p_x[0]: p_x[1]]
            if tmp_shape != (d, h, w):
                continue
            patches.append(tmp)
            labels.append(lbl)
            count += 1
        patches = numpy.array(patches)
        labels = numpy.array(labels)
        return patches, labels

    def norm_slices(self, slice_not):
        normed_slices = numpy.zeros((5, 146, 192, 152)).astype(numpy.float32)
        for slice_ix in range(4):
            normed_slices[slice_ix] = slice_not[slice_ix]
            for mode_ix in range(146):
                normed_slices[slice_ix][mode_ix] = self._normalize(slice_not[slice_ix][mode_ix])
        normed_slices[-1] = slice_not[-1]
        return normed_slices

    def _normalize(self, slice):
        b = numpy.percentile(slice, 99)
        t = numpy.percentile(slice, 1)
        slice = numpy.clip(slice, t, b)
        image_nonzero = slice[numpy.nonzero(slice)]
        if numpy.std(slice) == 0 or numpy.std(image_nonzero) == 0:
            return slice
        else:
            tmp = (slice - numpy.mean(image_nonzero)) / numpy.std(image_nonzero)
            tmp[tmp == tmp.min()] = -9
            return tmp


def dice(y_true, y_pred):
    sum_p = keras.backend.sum(y_pred, axis=0)
    sum_r = keras.backend.sum(y_true, axis=0)
    sum_pr =keras.backend.sum(y_true * y_pred, axis=0)
    dice_numerator = 2 * sum_pr
    dice_denominator = sum_r + sum_p
    dice_score = (dice_numerator + keras.backend.epsilon()) / (dice_denominator + keras.backend.epsilon())
    return dice_score


def dice_whole_metric(y_true, y_pred):
    y_true_f = keras.backend.reshape(y_true, shape=(-1, 4))
    y_pred_f = keras.backend.reshape(y_pred, shape=(-1, 4))
    y_whole = keras.backend.sum(y_true_f[:, 1:], axis=1)
    p_whole = keras.backend.sum(y_pred_f[:, 1:], axis=1)
    dice_whole = dice(y_whole, p_whole)
    return dice_whole


def dice_en_metric(y_true, y_pred):
    y_true_f = keras.backend.reshape(y_true, shape=(-1, 4))
    y_pred_f = keras.backend.reshape(y_pred, shape=(-1, 4))
    y_enh = y_true_f[:, -1]
    p_enh = y_pred_f[:, -1]
    dice_en = dice(y_enh, p_enh)
    return dice_en


def dice_core_metric(y_true, y_pred):
    y_true_f = keras.backend.reshape(y_true, shape=(-1, 4))
    y_pred_f = keras.backend.reshape(y_pred, shape=(-1, 4))
    y_core = keras.backend.sum(y_true_f[:, [1, 3]], axis=1)
    p_core = keras.backend.sum(y_pred_f[:, [1, 3]],axis=1)
    dice_core = dice(y_core, p_core)
    return dice_core


def weighted_log_loss(y_true, y_pred):
    y_pred /= keras.backend.sum(y_pred, axis=-1, keepdims=True)
    y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
    weights = numpy.array([1, 5, 2, 4])
    weights = keras.backend.variable(weights)
    loss = y_true * keras.backend.log(y_pred) * weights
    loss = keras.backend.mean(-keras.backend.sum(loss, -1))
    return loss


def gen_dice_loss(y_true, y_pred):
    y_true_f = keras.backend.reshape(y_true, shape=(-1, 4))
    y_pred_f = keras.backend.reshape(y_pred, shape=(-1, 4))
    sum_p = keras.backend.sum(y_pred_f, axis=-2)
    sum_r = keras.backend.sum(y_true_f, axis=-2)
    sum_pr = keras.backend.sum(y_true_f * y_pred_f, axis=-2)
    weights = keras.backend.pow(keras.backend.square(sum_r) + keras.backend.epsilon(), -1)
    generalized_dice_numerator = 2 * keras.backend.sum(weights * sum_pr)
    generalized_dice_denominator = keras.backend.sum(weights * (sum_r + sum_p))
    generalized_dice_score = generalized_dice_numerator / generalized_dice_denominator
    GDL = 1 - generalized_dice_score
    del sum_p, sum_r, sum_pr, weights
    return GDL + weighted_log_loss(y_true, y_pred)


class UnetModel(object):

    def __init__(self, img_shape, load_model_weights=None):
        self.img_shape = img_shape
        self.load_model_weights = load_model_weights
        self.model = self.compile_unet()

    def compile_unet(self):
        i = keras.layers.Input(shape=self.img_shape)
        i_ = keras.layers.GaussianNoise(stddev=0.01)(i)
        i_ = keras.layers.Conv2D(64, 2, padding='same', data_format='channels_last')(i_)
        out = self.unet(inputs=i_)
        model = keras.Model(input=i, output=out)
        sgd = keras.optimizers.SGD(lr=8e-2, momentum=9e-1, decay=5e-6, nesterov=False)
        model.compile(loss=gen_dice_loss, optimizer=sgd, metrics=[dice_whole_metric, dice_core_metric, dice_en_metric])
        if self.load_model_weights is not None:
            model.load_weights(self.load_model_weights)
        return model

    def unet(self, inputs, nb_classes=4, start_ch=64, depth=3, inc_rate=2.0, activation='relu', dropout=0.0, batchnorm=True, upconv=True, format_='channels_last'):
        o = self.level_block(inputs, start_ch, depth, inc_rate, activation, dropout, batchnorm, upconv, format_)
        o = keras.layers.BatchNormalization(o)
        o = keras.layers.PReLU(shared_axes=[1, 2])(o)
        o = keras.layers.Conv2D(nb_classes, 1, padding='same', data_format=format_)(o)
        o = keras.layers.Activation('softmax')(o)
        return o

    def level_block(self, m, dim, depth, inc, activ,do, bn, up, format_='channels_last'):
        if depth > 0:
            n = self.res_block_enc(m, 0.0, dim, activ, bn, format_)
            m = keras.layers.Conv2D(int(inc * dim), 2, strides=2, padding='same', data_format=format_)(n)
            m = self.level_block(m, int(inc * dim), depth-1, inc, activ, do, bn ,up)
            if up:
                m = keras.layers.UpSampling2D(size=(2, 2), data_format=format_)
                m = keras.layers.Conv2D(dim, 2, padding='same', data_format=format_)
            else:
                m = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', data_format=format_)(m)
            n = keras.layers.concatenate([n, m])
            m = self.res_block_dec(n, 0.0, dim, activ, bn, format_)
        else:
            m = self.res_block_enc(m, 0.0, dim, activ, bn, format_)
        return m

    def res_block_enc(self, m, dropout, dim, activ, bn, format_='channels_last'):
        n = keras.layers.BatchNormalization()(m) if bn else m
        n = keras.layers.PReLU(shared_axes=[1, 2])(n)
        n = keras.layers.Conv2D(dim, 3, padding='same', data_format=format_)(n)
        n = keras.layers.BatchNormalization()(n) if bn else n
        n = keras.layers.PReLU(shared_axes=[1, 2])(n)
        n = keras.layers.Conv2D(dim, 3, padding='same', data_format=format_)(n)
        n = keras.layers.add([m, n])
        return n

    def res_block_dec(self, m, dropout, dim, activ, bn, format_='channels_last'):
        n = keras.layers.BatchNormalization()(m) if bn else m
        n = keras.layers.PReLU(shared_axes=[1, 2])(n)
        n = keras.layers.Conv2D(dim, 3, padding='same', data_format=format_)(n)
        n = keras.layers.BatchNormalization()(n) if bn else n
        n = keras.layers.PReLU(shared_axes=[1, 2])(n)
        n = keras.layers.Conv2D(dim, 3, padding='same', data_format=format_)(n)
        save = keras.layers.Conv2D(dim, 1, padding='same', data_format=format_, use_bias=False)(m)
        n = keras.layers.add([save, n])
        return n


class SGDLearningRateTracker(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        optimizer = self.model.optimizer
        lr = keras.backend.get_value(optimizer.lr)
        decay = keras.backend.get_value(optimizer.decay)
        lr = lr / 10
        decay = decay * 10
        keras.backend.set_value(optimizer.lr, lr)
        keras.backend.set_value(optimizer.decay, decay)
        print('LR changed to:', lr)
        print('Decay changed to:', decay)


class Training(object):
    def __init__(self, batch_size, nb_epoch, load_model_resume_training=None):
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        if load_model_resume_training is not None:
            self.model = keras.models.load_model(load_model_resume_training, custom_objects={'gen_dice_loss': gen_dice_loss, 'dice_whole_metric': dice_whole_metric})
            print('pre-trained model loader')
        else:
            unet = UnetModel(img_shape=(128, 128, 4))
            self.model = unet.model
            print('U-net CNN compiled')

    def fit_unet(self, X33_train, Y_train, X_patches_valid=None, Y_labels_valid=None):
        train_generator = self.img_msk_gen(X33_train, Y_train, 9999)
        checkpointer = keras.callbacks.ModelCheckpoint(filepath='brain_segementation/ResUnet.{epoch:02d}_{val_loss:.3f}.hdf5', verbose=1)
        self.model.fit_generator(train_generator, steps_per_epoch=len(X33_train)//self.batch_size, epochs=self.nb_epoch, validation_data={X_patches_valid, Y_labels_valid}, verbose=1, callbacks=[checkpointer, SGDLearningRateTracker()])

    def img_msk_gen(self, X33_train, Y_train_, seed):
        datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, data_format='channels_last')
        datagen_msk = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, data_format='channels_last')
        image_generator = datagen.flow(X33_train, batch_size=4, seed=seed)
        y_generator = datagen_msk.flow(Y_train_, batch_size=4, seed=seed)
        while True:
            yield(image_generator.next(), y_generator.next())

    def save_model(self, model_name):
        model_tosave = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        json_string = self.model.to_json()
        self.model.save_weights(weights)
        with open(model_tosave, 'w') as f:
            json.dump(json_string, f)
        print('Model Saved')

    def load_model(self, model_name):
        print('Loading model {}'.format(model_name))
        model_toload = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        with open(model_toload) as f:
            m = next(f)
        model_comp = keras.models.model_from_json(json.load(m))
        model_comp.load_weights(weights)
        print('Model Loaded.')
        self.model = model_comp
        return model_comp


