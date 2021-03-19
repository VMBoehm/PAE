"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pae.load_data as ld
import tensorflow as tf
import numpy as np
import os
from functools import partial

load_funcs=dict(banana=ld.load_banana,mnist=ld.load_mnist, fmnist=ld.load_fmnist, cifar10=ld.load_cifar10, celeba=ld.load_celeba)

def add_noise(x,sigma=0.1):
    nn = tf.random.normal(tf.shape(input=x), dtype=tf.float32)
    x  = x+nn*sigma
    return x

#def rotate(x,max_ang=10.):
#    max_ang   = max_ang*np.pi/180.
#    shape     = tf.shape(input=x)
#    batchsize = shape[0]
#    square    = tf.math.reduce_prod(input_tensor=shape[1:])
#    onedim    = tf.cast(tf.math.sqrt(tf.cast(square,dtype=tf.float32)),dtype=tf.int32)
#    rot_ang = tf.random.uniform([batchsize],minval=-max_ang, maxval=max_ang)
#    x = tf.reshape(x, shape=[batchsize,onedim,onedim,1])
#    x = tf.contrib.image.rotate(x,rot_ang)
#    x = tf.reshape(x,shape)
#    return x

def build_input_fns(params,label,flatten,num_repeat=2):
    """Builds an iterator switching between train and heldout data."""

    print('loading %s dataset'%params['data_set'])

    load_func                            = partial(load_funcs[params['data_set']])
    x_train, _, x_test, _, x_valid, _    = load_func(params['data_dir'],flatten)
    #num_classes                          = len(np.unique(y_train))
    
    #if label in np.arange(num_classes):
    #    index   = np.where(y_train==label)
    #    x_train = x_train[index]
    #    y_train = y_train[index]
    #    index   = np.where(y_test==label)
    #    x_test  = x_test[index]
    #    y_test  = y_test[index]
    #elif label ==-1:
    #    pass
    #else:
    #    raise ValueError('invalid class')
    
    train_sample_size = len(x_train)
    test_sample_size  = len(x_test)

    x_train  = x_train.astype(np.float32)
    shape    = [params['batch_size']]+[ii for ii in x_train.shape[1:]]
    x_test   = x_test.astype(np.float32)

    def train_input_fn():
        def mapping_function(x):
            def extract_images(inds):
                return x_train[inds]/255.-0.5
            xx = tf.compat.v1.py_func(extract_images,[x],tf.float32)
            xx.set_shape(shape)
            return xx

        train_dataset  = tf.data.Dataset.range(train_sample_size)
        trainset       = train_dataset.shuffle(max(train_sample_size,10000)).repeat().batch(params['batch_size'],drop_remainder=True)
        trainset       = trainset.map(mapping_function)
        iterator = tf.compat.v1.data.make_one_shot_iterator(trainset)
        return iterator.get_next()

    def eval_input_fn():
        def mapping_function(x):
            def extract_images(inds):
                return x_test[inds]/255.-0.5
            xx = tf.compat.v1.py_func(extract_images,[x],tf.float32)
            xx.set_shape(shape)
            return xx

        test_dataset  = tf.data.Dataset.range(test_sample_size)
        testset       = test_dataset.shuffle(max(test_sample_size,10000)).repeat(num_repeat).batch(params['batch_size'],drop_remainder=True)
        testset       = testset.map(mapping_function)
        return tf.compat.v1.data.make_one_shot_iterator(testset).get_next()

    return train_input_fn, eval_input_fn


