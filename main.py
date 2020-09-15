"""
Copyright 2019 Vanessa Martina Boehm

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

# standard packages
from absl import flags
import numpy as np
import functools
import os
import pickle as pkl
import math
import sys
# tensorflow packages
import tensorflow as tf
import tensorflow_hub as hub

import pae.create_datasets as crd
from  pae.model import model_fn

flags.DEFINE_string('model_dir', default=os.path.join(os.path.abspath('./'),'model'), help='directory for storing the model (absolute path)')
flags.DEFINE_enum('data_set','mnist',['fmnist','cifar10','celeba','mnist','banana'], help='the tensorflow-dataset to load')
flags.DEFINE_string('module_dir', default=os.path.join(os.path.abspath('./'),'modules'), help='directory to which to export the modules (absolute path)')
flags.DEFINE_string('data_dir', default=os.path.join(os.path.abspath('./'),'data'), help='directory to store the data')
flags.DEFINE_integer('celeba_dim', default=64, help='pixel dimension of celeb data')

flags.DEFINE_float('learning_rate', default=1e-4, help='learning rate')    
flags.DEFINE_integer('batch_size',default=16, help='batch size')
flags.DEFINE_integer('max_steps', default=500000, help='training steps')    
flags.DEFINE_integer('n_steps', default=5000, help='number of training steps after which to perform the evaluation')
flags.DEFINE_enum('loss', 'AE', ['VAE','hybrid','AE'] , help='which objective to optimize')
flags.DEFINE_boolean('output_images', default=True, help='whether to output image summaries')
flags.DEFINE_boolean('full_sigma', default=True, help='whether to use constant or pixel-wise noise')
flags.DEFINE_boolean('sigma_annealing', default=False, help='whether to run a scheduled beta annealing on the KL term (VAE only)')
flags.DEFINE_boolean('beta_VAE', default=True,help='whether to run a beta VAE')
flags.DEFINE_float('beta',default=120,help='beta paramater for beta VAE')
flags.DEFINE_boolean('free_bits', default=False, help='whether to train a VAE with free bits')
flags.DEFINE_float('lambda', default=0, help='free bits parameter')
flags.DEFINE_boolean('C_annealing', default=True, help='whether to reduce available kl with training')
flags.DEFINE_float('C', default=18, help='C parameter')
flags.DEFINE_spaceseplist('augmentation', ['rot'], 'data augmentation types. Must be one or a list of the following: None, rot, flip, crop, bright')
flags.DEFINE_float('rot_angle', 5., 'maximum rotation in degrees for data augmentation') 

flags.DEFINE_integer('latent_size',default=10, help='dimensionality of latent space')
flags.DEFINE_string('activation', default='tanh', help='activation function')
flags.DEFINE_integer('n_samples', default=16, help='number of samples for encoding')
flags.DEFINE_enum('network_type', 'vae10', ['vae10','fully_connected','conv', 'infoGAN','resnet_fc','resnet_conv'], help='which type of network to use, currently supported: fully_conneted and conv')
flags.DEFINE_integer('n_filt',default=32,help='number of filters to use in the first convolutional layer')
flags.DEFINE_integer('dense_size', default=256, help='number of connnections in the fc resnet')
flags.DEFINE_integer('n_layers',default=4, help='number of layers in the fc resnet')
flags.DEFINE_boolean('bias', default=False, help='whether to use a bias in the convolutions')
flags.DEFINE_float('dropout_rate', default=0, help='dropout rate used in infoGAN')
flags.DEFINE_float('sigma', default=0.1, help='initial value of sigma in Gaussian likelihood')
flags.DEFINE_integer('class_label', default=-1, help='number of specific class to train on. -1 for all classes')
flags.DEFINE_string('tag', default='test', help='optional additional tag that is added to name of the run')


#conv resnet flags
flags.DEFINE_integer('base_dim', default=16, help='after first convolutional layer in resnet')
flags.DEFINE_integer('kernel_size', default=3, help='kernel size of convolutional layers in resnet')
flags.DEFINE_integer('num_scale', default=4, help='number of scaling blocks in resnet')
flags.DEFINE_integer('block_per_scale', default=1, help='number of residual blocks per scaling block')
flags.DEFINE_integer('depth_per_block', default=2, help='depth of network in each block')
flags.DEFINE_integer('fc_dim', default=512, help='output dimensionality of fully conncted residual block')

FLAGS= flags.FLAGS


def main(argv):
    del argv
    DATA_SHAPES = dict(mnist=[28,28,1],fmnist=[28,28,1],cifar10=[32,32,3],celeba=[FLAGS.celeba_dim,FLAGS.celeba_dim,3],banana=[32,1])
    params = FLAGS.flag_values_dict()
    DATA_SHAPE = DATA_SHAPES[FLAGS.data_set]
    params['activation']  = getattr(tf.nn, params['activation'])
    print(DATA_SHAPE)
    if len(DATA_SHAPE)>2:
        params['width']       = DATA_SHAPE[0]
        params['height']      = DATA_SHAPE[1]
        params['n_channels']  = DATA_SHAPE[2]
    else:
        params['length']      = DATA_SHAPE[0]
        params['n_channels']  = DATA_SHAPE[1]
    
    params['data_shape'] = DATA_SHAPE
    flatten = True

    params['output_size'] = np.prod(DATA_SHAPE)
    params['full_size']   = [params['batch_size'],params['output_size']] 

    if params['network_type'] in ['conv','infoGAN','resnet_conv']:
        flatten = False
        params['output_size'] = DATA_SHAPE
        params['full_size']   = [params['batch_size'],params['width'],params['height'],params['n_channels']]

    if params['full_sigma']:
        params['tag']+='_full_sigma'
    else:
        params['tag']+='_mean_sigma'

    if params['beta']:
        if params['loss']=='VAE':
            params['tag']+='_beta%d'%params['beta']
    if params['C_annealing']:
        if params['loss']=='VAE':
            params['tag']+='_C%d'%params['C']
    
    params['label']       = os.path.join('%s'%params['data_set'], 'class%d'%params['class_label'], 'latent_size%d'%params['latent_size'],'net_type_%s'%params['network_type'],'loss_%s'%params['loss'],params['tag'])

    params['model_dir']   = os.path.join(params['model_dir'], params['label'])
    params['module_dir']  = os.path.join(params['module_dir'], params['label'])
    print(params['module_dir'])
    
    for dd in ['model_dir', 'module_dir', 'data_dir']:
        if not os.path.isdir(params[dd]):
            os.makedirs(params[dd], exist_ok=True)

    if not os.path.isdir('./params'):
        os.makedirs('./params')
    pkl.dump(params, open('./params/params_%s_%d_%d_%s_%s_%s.pkl'%(params['data_set'],params['class_label'],params['latent_size'],params['network_type'],params['loss'],params['tag']),'wb'))
 
    train_input_fn, eval_input_fn = crd.build_input_fns(params,label=FLAGS.class_label,flatten=flatten)

    estimator = tf.estimator.Estimator(model_fn, params=params, config=tf.estimator.RunConfig(model_dir=params['model_dir']))
    c         = tf.placeholder(tf.float32,params['full_size'])
    serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=dict(x=c))

    exporter   = hub.LatestModuleExporter("tf_hub", serving_input_fn)
    #lr = params['learning_rate']
    n_epoch = 0 
    n_steps = FLAGS.n_steps
    for ii in range(FLAGS.max_steps//n_steps):	
        #params['learning_rate'] = lr * math.pow(0.5, np.floor(float(n_epoch) / float(150)))
        estimator.train(train_input_fn, steps=n_steps)
        eval_results = estimator.evaluate(eval_input_fn)
        print('model evaluation on test set:', eval_results)
        print('n_epoch', n_epoch)
        exporter.export(estimator, params['module_dir'], estimator.latest_checkpoint())
    return True

if __name__ == "__main__":
    tf.app.run()
