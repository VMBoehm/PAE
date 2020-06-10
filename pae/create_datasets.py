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
import tensorflow_datasets as tfds
from functools import partial
import scipy.ndimage as ndimage


load_funcs=dict(banana=ld.load_banana,mnist=ld.load_mnist, fmnist=ld.load_fmnist, cifar10=ld.load_cifar10, celeba=ld.load_celeba)

def dequantize(x):
    nn = np.random.uniform(size=x.shape)
    x  = x+nn
    return x.astype(np.float32)

def random_rotate_image(image,params,flatten):
    if flatten:
        image = image.reshape((params['batch_size'],params['width'],params['height']))
    image = ndimage.rotate(image, np.random.uniform(-params['rot_angle'], params['rot_angle']),axes=(2,1),reshape=False)
    if flatten:
        image = image.reshape((params['batch_size'],-1))
    return image



def build_input_fns(params,label,flatten):
    """Builds an iterator switching between train and heldout data."""

    print('loading %s dataset'%params['data_set'])

    load_func                            = partial(load_funcs[params['data_set']])
    x_train, _, x_test, _, x_valid, _    = load_func(params['data_dir'],flatten,params)
    
    train_sample_size = len(x_train)
    test_sample_size  = len(x_test)

    x_train  = x_train.astype(np.float32)
    shape    = [params['batch_size']]+[ii for ii in x_train.shape[1:]]
    x_test   = x_test.astype(np.float32)

    def augment(image):
        if 'crop' in params['augmentation']:
            image = tf.image.random_crop(image, [params['batch_size'],params['width']-6,params['height'],params['n_channels']]) 
            image = tf.image.resize_images(image, size=[params['width'],params['height']]) 
        if 'bright' in params['augmentation']:
            image = tf.image.random_brightness(image, max_delta=0.2) # Random brightness
        if 'flip' in params['augmentation']:
            image = tf.image.random_flip_left_right(image)
        return image


    def train_input_fn():
        def mapping_function(x):
            def extract_images(inds):
                xx = dequantize(x_train[inds])
                if 'rot' in params['augmentation']:
                    xx = random_rotate_image(xx,params,flatten)
                return xx/255.-0.5
            xx = tf.py_func(extract_images,[x],tf.float32)
            xx.set_shape(shape)
            return xx

        train_dataset  = tf.data.Dataset.range(train_sample_size)
        trainset       = train_dataset.shuffle(max(train_sample_size,10000)).repeat().batch(params['batch_size'],drop_remainder=True)
        trainset       = trainset.map(mapping_function)
        trainset       = trainset.map(augment)
        iterator = tf.compat.v1.data.make_one_shot_iterator(trainset)
        return iterator.get_next()

    def eval_input_fn():
        def mapping_function(x):
            def extract_images(inds):
                xx = dequantize(x_test[inds])
                return xx/255.-0.5
            xx = tf.py_func(extract_images,[x],tf.float32)
            xx.set_shape(shape)
            return xx

        test_dataset  = tf.data.Dataset.range(test_sample_size)
        testset       = test_dataset.shuffle(max(test_sample_size,10000)).repeat(2).batch(params['batch_size'],drop_remainder=True)
        testset       = testset.map(mapping_function)
        return tf.compat.v1.data.make_one_shot_iterator(testset).get_next()

    return train_input_fn, eval_input_fn


def build_input_fn_celeba(params):
    """
    Creates input functions for training, validation, and testing
    """

    def input_fn(is_training=False, tag='train', shuffle_buffer=10000):

        data_dir = os.path.join(params['data_dir'],'celeba/')

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        celeba_builder = tfds.builder('celeb_a')
        # Download the dataset
        celeba_builder.download_and_prepare()
        # Construct a tf.data.Dataset
        data = celeba_builder.as_dataset(split=tag)

        dset = data#[tag]
        # Crop celeb image and resize
        dset = dset.map(lambda x: tf.cast(tf.image.resize_image_with_crop_or_pad(x['image'], 128, 128), tf.float32) / 256.)

        if is_training:
            dset = dset.repeat()
            dset = dset.map(lambda x: tf.image.random_flip_left_right(x),num_parallel_calls=2)
            dset = dset.shuffle(buffer_size=shuffle_buffer)

        dset = dset.batch(params['batch_size'],drop_remainder=True)
        dset = dset.map(lambda x: tf.image.resize_bilinear(x, (64, 64)),num_parallel_calls=2)
        dset = dset.prefetch(params['batch_size'])
        return dset

        return input_fn

    return {'train':partial(input_fn, tag='train', is_training=True),
            'validation': partial(input_fn, tag='validation'),
            'test': partial(input_fn, tag='test')}