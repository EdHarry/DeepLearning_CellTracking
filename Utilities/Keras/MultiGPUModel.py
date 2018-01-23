import tensorflow as tf
from keras import backend as keras_bkend
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Lambda

def slice_batch(x, n_gpu, part):
    sh = keras_bkend.shape(x)
    L = sh[0] // n_gpu
    if part == n_gpu - 1:
        result = x[part*L:]
    else:
        result = x[part*L:(part+1)*L]
    
    return result

def to_multi_gpu(model, n_gpu=2):
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name=model.input_names[0])
    towers = []
    for gpu in range(n_gpu):
        with tf.device('/gpu:' + str(gpu)):
            slice_gpu = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpu':n_gpu, 'part':gpu})(x)
            towers.append(model(slice_gpu))
    with tf.device('/cpu:0'):
        merged = merge(towers, mode='concat', concat_axis=0)

    return Model(input=[x], output=merged)