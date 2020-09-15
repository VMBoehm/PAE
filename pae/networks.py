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

import tensorflow as tf
import tensorflow_hub as hub

from pae.util_2stageVAE import *


#### from two-stage VAE ####

def res_block(x, out_dim, is_training, name, depth=2, kernel_size=3):
    with tf.variable_scope(name):
        y = x
        for i in range(depth):
            y = tf.nn.relu(batch_norm(y, is_training, 'bn'+str(i)))
            y = tf.layers.conv2d(y, out_dim, kernel_size, padding='same', name='layer'+str(i))
        s = tf.layers.conv2d(x, out_dim, kernel_size, padding='same', name='shortcut')
        return y + s 


def res_fc_block(x, out_dim, name, depth=2):
    with tf.variable_scope(name):
        y = x 
        for i in range(depth):
            y = tf.layers.dense(tf.nn.relu(y), out_dim, name='layer'+str(i))
        s = tf.layers.dense(x, out_dim, name='shortcut')
        return y + s 


def scale_block(x, out_dim, is_training, name, block_per_scale=1, depth_per_block=2, kernel_size=3):
    with tf.variable_scope(name):
        y = x 
        for i in range(block_per_scale):
            y = res_block(y, out_dim, is_training, 'block'+str(i), depth_per_block, kernel_size)
        return y 


def scale_fc_block(x, out_dim, name, block_per_scale=1, depth_per_block=2):
    with tf.variable_scope(name):
        y = x 
        for i in range(block_per_scale):
            y = res_fc_block(y, out_dim, 'block'+str(i), depth_per_block)
        return y 

def downsample(x, out_dim, kernel_size, name):
    with tf.variable_scope(name):
        input_shape = x.get_shape().as_list()
        assert(len(input_shape) == 4)
        return tf.layers.conv2d(x, out_dim, kernel_size, 2, 'same')
#### ----------------####

def resnet_encoder(params, is_training):

    is_training = tf.constant(is_training, dtype=tf.bool)

    def encoder(x):
        with tf.variable_scope('model/encoder',['x'], reuse=tf.AUTO_REUSE):
            dim = params['base_dim'] 
            net = tf.layers.conv2d(x, dim, params['kernel_size'], 1, 'same', name='conv0')
            for i in range(params['num_scale']):
                net = scale_block(net, dim, is_training, 'scale'+str(i), params['block_per_scale'], params['depth_per_block'], params['kernel_size'])
                if i != params['num_scale'] - 1:
                    dim *= 2
                    net = downsample(net, dim, params['kernel_size'], 'downsample'+str(i))#downsamples by a factor of 2
            net = tf.reduce_mean(net, [1, 2])
            net = scale_fc_block(net, params['fc_dim'], 'fc', 1, params['depth_per_block'])
            net = tf.layers.dense(net, 2*params['latent_size'])
        return net

    return encoder

def resnet_decoder(params, is_training):
    
    is_training = tf.constant(is_training, dtype=tf.bool)
    
    desired_scale = params['data_shape'][0]
    print(desired_scale)
    scales, dims  = [], []
    current_scale, current_dim = 2, params['base_dim'] 
    while current_scale <= desired_scale:
        scales.append(current_scale)
        dims.append(current_dim)
        current_scale*=2
        current_dim = min(current_dim*2, 1024)
    assert(scales[-1] == desired_scale)
    dims = list(reversed(dims))
    
    def decoder(z):
        with tf.variable_scope('model/decoder',['z'],reuse=tf.AUTO_REUSE):
            data_depth = params['data_shape'][-1]
            fc_dim     = 2 * 2 * dims[0]
            net        = tf.layers.dense(z, fc_dim, name='fc0')
            net        = tf.reshape(net, [-1, 2, 2, dims[0]])
            for i in range(len(scales)-1):
                net = upsample(net, dims[i+1], params['kernel_size'], 'up'+str(i))
                net = scale_block(net, dims[i+1], is_training, 'scale'+str(i), params['block_per_scale'], params['depth_per_block'], params['kernel_size'])
            net = tf.layers.conv2d(net, data_depth, params['kernel_size'], 1, 'same')
            net = tf.nn.sigmoid(net)
            return net-0.5
    return decoder

#############################
def mnist_resnet_decoder(params,is_training, reuse=tf.AUTO_REUSE):
    """
    Decoder using a dense neural network
    """
    def decoder(z):
        with tf.variable_scope('model/decoder', ['z'], reuse=reuse):
            net = tf.layers.dense(z, params['dense_size'], activation=None)

            for i in range(params['n_layers']):
                shortcut = net
                net = tf.layers.dense(net, params['dense_size'], activation=params['activation'])
                net = tf.layers.dense(net, params['dense_size'], activation=params['activation'])
                net = net + shortcut

            net = tf.layers.dense(net, params['output_size'], activation=None)
        return net

    return decoder


def mnist_resnet_encoder(params,is_training, reuse=tf.AUTO_REUSE):
    
    def encoder(x):
        with tf.variable_scope('model/encoder', ['x'], reuse=reuse):
           net = tf.layers.flatten(x)
           net = tf.layers.dense(net, params['dense_size'], activation=None)

        for i in range(params['n_layers']):
            shortcut = net
            net = tf.layers.dense(net, params['dense_size'], activation=params['activation'])
            net = tf.layers.dense(net, params['dense_size'], activation=params['activation'])
            net = net + shortcut

        net = tf.layers.dense(net, 2*params['latent_size'], activation=None)
        return net

    return encoder


def infoGAN_encoder(params,is_training):

    is_training = tf.constant(is_training, dtype=tf.bool)

    def encoder(x):
        with tf.variable_scope('model/encoder',['x'], reuse=tf.AUTO_REUSE):  

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='conv1', use_sn=True))
            net = conv2d(net, 128, 4, 4, 2, 2, name='conv2', use_sn=True)
            net = batch_norm(net, is_training=is_training, scope='b_norm1')
            net = tf.layers.dropout(net,rate=params['dropout_rate'],training=is_training)
            net = lrelu(net)
    
            net = tf.reshape(net, [params['batch_size'], -1])
            net = linear(net, 1024, scope="ln1", use_sn=True)
            net = batch_norm(net, is_training=is_training, scope='b_norm2')
            net = tf.layers.dropout(net,rate=params['dropout_rate'],training=is_training)
            net = lrelu(net)

            net = linear(net, 2 * params['latent_size'], scope="ln_output", use_sn=True)
        
        return net

    return encoder


def infoGAN_decoder(params,is_training):

    is_training = tf.constant(is_training, dtype=tf.bool)

    def decoder(z):
        with tf.variable_scope('model/decoder',['z'], reuse=tf.AUTO_REUSE):
        
            net = tf.nn.relu(batch_norm(linear(z, 1024, 'ln2'), is_training=is_training, scope='b_norm3'))
            net = tf.nn.relu(batch_norm(linear(net, 128 * (params['width'] // 4) * (params['height'] // 4), scope='ln3'), is_training=is_training, scope='b_norm4'))
            net = tf.layers.dropout(net,rate=params['dropout_rate'],training=is_training)
            net = tf.reshape(net, [params['batch_size'], params['width'] // 4, params['height'] // 4, 128])
            net = tf.nn.relu(batch_norm(deconv2d(net, [params['batch_size'], params['width'] // 2, params['height'] // 2, 64], 4, 4, 2, 2, name='conv3'), is_training=is_training, scope='b_norm5'))
            net = tf.layers.dropout(net,rate=params['dropout_rate'],training=is_training) 
            net = tf.nn.sigmoid(deconv2d(net, [params['batch_size'], params['width'], params['height'], params['n_channels']], 4, 4, 2, 2, name='conv4'))
            net = net-0.5
        return net

    return decoder


def conv_encoder(params, is_training=True):

    activation  = params['activation']
    latent_size = params['latent_size']
    n_filt      = params['n_filt']
    bias        = params['bias']
    dataset     = params['data_set']

    def encoder(x):
        with tf.variable_scope('model/encoder',['x'], reuse=tf.AUTO_REUSE):

            net = tf.layers.conv2d(x,n_filt,5,strides=2,activation=None, padding='SAME', use_bias=bias) #64x64 -> 32x32/ 28x28 -> 14x14
            net = tf.layers.batch_normalization(net, training=is_training)
            net = activation(net)

            net = tf.layers.conv2d(net, n_filt*2, 3, 2, activation=None, padding='SAME', use_bias=bias) #32x32 -> 16x16 / 14x14 -> 7*7
            net = tf.layers.batch_normalization(net, training=is_training)
            net = activation(net)

            if dataset in ['celeba']:
                net = tf.layers.conv2d(net, n_filt*4, 5, 2, activation=None, padding='SAME', use_bias=bias) #16x16 -> 8x8 
                net = tf.layers.batch_normalization(net, training=is_training)
                net = activation(net)

            net = tf.layers.flatten(net) #8x8*n_filt*4
            net = tf.layers.dense(net, latent_size*2, activation=None)
            return net

    return encoder


def conv_decoder(params, is_training=True):

    activation = params['activation']
    latent_size= params['latent_size']
    n_filt     = params['n_filt']
    bias       = params['bias']
    dataset    = params['data_set']

    def decoder(z):
        with tf.variable_scope('model/decoder',['z'], reuse=tf.AUTO_REUSE):

            if dataset in ['celeba','cifar10']:
                NN = 8
            elif dataset in ['mnist','fmnist']: 
                NN = 7
            net = tf.layers.dense(z,n_filt*4*NN*NN,activation=activation, use_bias=bias)
            net = tf.reshape(net, [-1, NN, NN,n_filt*4])

            if dataset in ['celeba']:
                net = tf.layers.conv2d_transpose(net,n_filt*4, 5, strides=2, padding='SAME', use_bias=bias) # output_size 16x16/14x14
                net = tf.layers.batch_normalization(net, training=is_training)
                net = activation(net)

            net = tf.layers.conv2d_transpose(net,n_filt*2, 3, strides=2, padding='SAME', use_bias=bias) # output_size 16x16/14x14 
            net = tf.layers.batch_normalization(net, training=is_training)
            net = activation(net)

            net = tf.layers.conv2d_transpose(net, n_filt, 5, strides=2, padding='SAME', use_bias=bias) # output_size 32x32/28x28
            net = tf.layers.batch_normalization(net, training=is_training)
            net = activation(net)

            net = tf.layers.conv2d_transpose(net, params['output_size'][-1], kernel_size=4, strides=1, activation=None, padding='same', name='output_layer')# bring to correct number of channels
        return net

    return decoder


def fully_connected_encoder(params,is_training):

    activation = params['activation']
    latent_size = params['latent_size']

    def encoder(x):
        with tf.variable_scope('model/encoder', reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(x, 512, name='dense_1', activation=activation)
            net = tf.layers.dense(net, 256, name='dense_2', activation=activation)
            net = tf.layers.dense(net, 128, name='dense_3', activation=activation)
            net = tf.layers.dense(net, 2*latent_size, name='dense_4', activation='sigmoid')
        return net
    return encoder 

def fully_connected_decoder(params,is_training):
    
    activation = params['activation']
    latent_size= params['latent_size']

    def decoder(z):
        with tf.variable_scope('model/decoder', reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(z, 128, name ='dense_1', activation=activation)
            net = tf.layers.dense(net, 256, name='dense_2', activation=activation)
            net = tf.layers.dense(net, 512, name='dense_3', activation=activation)
            net = tf.layers.dense(net, params['output_size'] , name='dense_4', activation='sigmoid')
        return net-0.5
    return decoder


def vae10_decoder(params,is_training):

    activation = params['activation']
    latent_size= params['latent_size']

    def decoder(z):
        with tf.compat.v1.variable_scope('model/decoder', reuse=tf.compat.v1.AUTO_REUSE):
            net = tf.compat.v1.layers.dense(z, 64, name ='dense_1', activation=activation)
            net = tf.compat.v1.layers.dense(net, 256, name='dense_2', activation=activation)
            net = tf.compat.v1.layers.dense(net, 256, name='dense_3', activation=activation)
            net = tf.compat.v1.layers.dense(net, 1024, name='dense_4', activation=activation)
            net = tf.compat.v1.layers.dense(net, params['output_size'] , name='dense_5', activation='sigmoid')
    return decoder

def vae10_encoder(params, is_training):
    
    activation = params['activation']
    latent_size = params['latent_size']
    
    def encoder(x):
        with tf.compat.v1.variable_scope('model/encoder', reuse=tf.compat.v1.AUTO_REUSE):
            net = tf.compat.v1.layers.dense(x, 256, name='dense_1', activation=activation)
            net = tf.compat.v1.layers.dense(net,64, name='dense_2', activation=activation)
            net = tf.compat.v1.layers.dense(net,2*latent_size, name='dense_3') 
    return network



def make_encoder(params, is_training):
    
    network_type = params['network_type']

    if network_type=='fully_connected':
        encoder_ = fully_connected_encoder(params,is_training)
    elif network_type=='conv':
        encoder_ = conv_encoder(params,is_training)
    elif network_type=='infoGAN':
        encoder_ = infoGAN_encoder(params,is_training)
    elif network_type=='resnet_conv':
        encoder_ = resnet_encoder(params, is_training)
    elif network_type=='resnet_fc':
        encoder_ = mnist_resnet_encoder(params, is_training)
    elif network_type=='vae10':
        encoder_ = vae10_encoder(params, is_training)
    else:
        raise NotImplementedError("Network type not implemented.")

    def encoder_spec():
        x = tf.placeholder(tf.float32, shape=params['full_size'])
        z = encoder_(x)
        hub.add_signature(inputs={'x':x},outputs={'z':z})

    enc_spec  = hub.create_module_spec(encoder_spec)

    encoder   = hub.Module(enc_spec, name='encoder',trainable=True)

    hub.register_module_for_export(encoder, "encoder")

    return encoder


def make_decoder(params,is_training):

    network_type = params['network_type']

    if network_type=='fully_connected':
        decoder_ = fully_connected_decoder(params,is_training)
    elif network_type=='conv':
        decoder_ = conv_decoder(params, is_training)
    elif network_type=='infoGAN':
        decoder_ = infoGAN_decoder(params, is_training)
    elif network_type=='resnet_conv':
        decoder_ = resnet_decoder(params, is_training)
    elif network_type=='resnet_fc':
        decoder_ = mnist_resnet_decoder(params, is_training)
    elif network_type=='vae10':
        decoder_ = vae10_decoder(params, is_training)
    else:
        raise NotImplementedError("Network type not implemented.")

    def decoder_spec():
        z = tf.placeholder(tf.float32, shape=[None,params['latent_size']]) 
        x = decoder_(z)
        hub.add_signature(inputs={'z':z},outputs={'x':x})

    dec_spec  = hub.create_module_spec(decoder_spec)

    decoder   = hub.Module(dec_spec, name='decoder',trainable=True)

    hub.register_module_for_export(decoder, "decoder")

    return decoder

