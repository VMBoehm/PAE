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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import pae.networks_tf2 as nw


### these two functions are inspired by https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py and modified to work with flattened data by adding a shape keyword

def pack_images(images, rows, cols,shape):
    """Helper utility to make a field of images.
    Borrowed from Tensorflow Probability
    """
def make_images(images, nrows, ncols,shape):
    width  = shape[-3]
    height = shape[-2]
    depth  = shape[-1]
    bsize  = tf.shape(input=images)[0]
    images = tf.reshape(images, (-1, width, height, depth))
    nrows  = tf.minimum(nrows, bsize)
    ncols  = tf.minimum(bsize//nrows, ncols)
    images = images[:nrows * ncols]
    images = tf.reshape(images, (nrows, ncols, width, height, depth))
    images = tf.transpose(a=images, perm=[0, 2, 1, 3, 4])
    images = images+0.5
    images = tf.clip_by_value(tf.reshape(images, [1, nrows * width, ncols * height, depth]), 0, 1)
    return images


def image_tile_summary(name, tensor, rows, cols, shape):
    tf.compat.v1.summary.image(name, make_images(tensor, rows, cols, shape), max_outputs=1)
#######

def get_prior(latent_size):
    return tfd.MultivariateNormalDiag(tf.zeros(latent_size), scale_identity_multiplier=1.0)

def get_GN_covariance(decoded,z,params):

    ones  = tf.linalg.eye(params['latent_size'], batch_shape=[params['batch_size']],dtype=tf.float32) 
    
    with tf.compat.v1.variable_scope("likelihood", reuse=tf.compat.v1.AUTO_REUSE):
        sigma = tf.compat.v1.get_variable(name='sigma', use_resource=False, initializer=tf.ones([np.prod(params['output_size'])])*params['sigma'])
    decoded = tf.reshape(decoded,[params['batch_size'],-1])
    grad_g  = tf.gather(tf.gradients(ys=decoded/sigma,xs=z),0)
    grad_g2 = tf.einsum('ij,ik->ijk',grad_g,grad_g)
    GNhess  = ones#+grad_g2
    cov     = tf.linalg.inv(GNhess)
    cov     = 0.5*(cov+tf.linalg.matrix_transpose(cov))
    det     = tf.linalg.det(cov)

#    hess    = tf.hessians(decoded,z)
#    hess    = tf.gather(hess, 0)
#    hess    = tf.reduce_sum(hess, axis = 2 )
#    hess    = 0.5*(hess+tf.linalg.transpose(hess))
#    detC    = 1./(tf.linalg.det(hess))

    return cov, det#, detC



def get_posterior(encoder):

    def posterior(x):
        mu, sigma        = tf.split(encoder({'x':x},as_dict=True)['z'], 2, axis=-1)
        sigma            = tf.nn.softplus(sigma) + 1e-6
        approx_posterior = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return approx_posterior

    return posterior



def get_likelihood(decoder,params):

    with tf.compat.v1.variable_scope("likelihood", reuse=tf.compat.v1.AUTO_REUSE):
        sigma = tf.compat.v1.get_variable(name='sigma', use_resource=False, initializer=tf.ones([np.prod(params['output_size'])])*params['sigma'])

    def likelihood(z):
        mean = decoder({'z':z},as_dict=True)['x']
        mean = tf.reshape(mean,[params['batch_size'],-1])
        return tfd.Independent(tfd.MultivariateNormalDiag(loc=mean,scale_diag=sigma))
        
    return likelihood


def get_laplace_posterior(z,decoder,params):

    decoded          = decoder({'z':z},as_dict=True)['x']
    cov, det         = get_GN_covariance(decoded,z,params)
    laplace_posterior= tfd.MultivariateNormalFullCovariance(loc=z,covariance_matrix=cov)

    return laplace_posterior, det

def model_fn(features, labels, mode, params, config):
    del labels, config
    try:
        features = features['x']
    except:
        pass

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    encoder          = nw.make_encoder(params, is_training)
    decoder          = nw.make_decoder(params, is_training)
    
    global_step      = tf.compat.v1.train.get_or_create_global_step()
    stage            = tf.greater(global_step,tf.constant(params['max_steps']//2,dtype=tf.int64))
    stage2           = tf.equal(global_step,tf.constant(params['max_steps']//2,dtype=tf.int64))

    prior            = get_prior(params['latent_size'])

    posterior        = get_posterior(encoder)
    approx_posterior = posterior(features)
    MAP              = approx_posterior.mean()

    decoded          = decoder({'z':MAP},as_dict=True)['x']

    chi2             = tf.stop_gradient(tf.reshape(tf.square(decoded-features),[params['batch_size'],-1]))
    
    #first taking sqrt and then averaging over batch (correct?)    	
    #sigma_pixel      = tf.reduce_mean(tf.sqrt(chi2),axis=0)
    #sigma_mean       = tf.reduce_mean(tf.sqrt(tf.reduce_mean(chi2,axis=1)),axis=0)
    #sigma_mean       = sigma_mean*tf.ones([np.prod(params['output_size'])])

    #first averaging over batch then sqrt      
    sigma_pixel      = tf.sqrt(tf.reduce_mean(input_tensor=chi2,axis=0))
    ema              = tf.train.ExponentialMovingAverage(decay=0.99)
    ema.apply([sigma_pixel])
    
    sigma_mean       = tf.sqrt(tf.reduce_mean(input_tensor=chi2))
    sigma_mean       = sigma_mean*tf.ones([np.prod(params['output_size'])])

    sigma_est        = tf.cond(pred=tf.constant(params['full_sigma'],dtype=tf.bool),true_fn=lambda: sigma_pixel,false_fn=lambda: sigma_mean)
    sigma_reg        = tf.maximum(sigma_est,params['sigma']*tf.ones([np.prod(params['output_size'])]))

    with tf.compat.v1.variable_scope("likelihood", reuse=tf.compat.v1.AUTO_REUSE):
        sigma = tf.compat.v1.get_variable(name='sigma',use_resource=False, initializer=tf.ones([np.prod(params['output_size'])])*params['sigma'])
    
    if not params['loss']=='VAE':
        sigma        = tf.cond(pred=stage2,true_fn=lambda:sigma.assign(sigma_reg),false_fn=lambda:sigma)
    
    likelihood       = get_likelihood(decoder,params)

#    z                = tf.Variable(tf.zeros([params['batch_size'],params['latent_size']]))
#    z                = z.assign(MAP)
#    laplace_posterior, det = get_laplace_posterior(z,decoder,params) 
#    kl_laplace_prior = tfd.kl_divergence(laplace_posterior, prior)
#    kl_laplace_vmf   = tfd.kl_divergence(laplace_posterior, approx_posterior)
    
    map_likelihood   = likelihood(MAP).log_prob(tf.reshape(features,[params['batch_size'],-1]))
    map_prior        = prior.log_prob(MAP)
    
    kl               = tfd.kl_divergence(approx_posterior, prior)

    posterior_sample  = approx_posterior.sample()
    sample_likelihood = likelihood(posterior_sample).log_prob(tf.reshape(features,[params['batch_size'],-1]))
        

    objective_AE      = map_likelihood
    objective_MAP     = map_prior+map_likelihood
    
    objective_VAE     = sample_likelihood-kl 

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:

        tf.compat.v1.summary.scalar('log_likelihood_at_MAP',tf.reduce_mean(input_tensor=map_likelihood))
        tf.compat.v1.summary.scalar('log_prior_at_MAP', tf.reduce_mean(input_tensor=map_prior))
        tf.compat.v1.summary.scalar('KL_divergence',tf.reduce_mean(input_tensor=kl))
        tf.compat.v1.summary.scalar('elbo',tf.reduce_mean(input_tensor=objective_VAE))
#        tf.summary.scalar('KL_laplace_prior', tf.reduce_mean(kl_laplace_prior))
#        tf.summary.scalar('KL_laplace_vmf', tf.reduce_mean(kl_laplace_vmf))
#        tf.summary.scalar('Gauss_Newton_determinant', tf.reduce_mean(det))
#        tf.summary.scalar('full_Hessian_determinant', tf.reduce_mean(detC))
        tf.compat.v1.summary.scalar('sigma', tf.reduce_mean(input_tensor=sigma))
        tf.compat.v1.summary.scalar('stage', tf.cast(stage, tf.int32))

        if params['loss']=='VAE':
            loss         = tf.cond(pred=stage,true_fn=lambda:-tf.reduce_mean(input_tensor=objective_VAE),false_fn=lambda:-tf.reduce_mean(input_tensor=objective_VAE))
        else:
            loss         = tf.cond(pred=stage,true_fn=lambda:-tf.reduce_mean(input_tensor=objective_MAP),false_fn=lambda:-tf.reduce_mean(input_tensor=objective_AE))
 
        all_vars         = tf.compat.v1.trainable_variables()
        train_vars       = [var for var in all_vars if 'sigma' not in var.name]

        if params['loss']=='VAE':
            lr               = tf.compat.v1.train.cosine_decay(params['learning_rate'],global_step,params['max_steps'],alpha=0.1)
        else:
            lr               = tf.compat.v1.train.cosine_decay_restarts(params['learning_rate'],global_step,params['max_steps']//2,t_mul=2.0,m_mul=1.0,alpha=0.1)
        
        tf.compat.v1.summary.scalar('learning_rate',lr)

        optimizer        = tf.compat.v1.train.AdamOptimizer(lr)

        train_op         = optimizer.minimize(loss,var_list=train_vars,global_step=global_step)

        if params['output_images']:
            if mode == tf.estimator.ModeKeys.TRAIN:
                image_tile_summary('training/inputs',features, rows=4, cols=4, shape=params['data_shape'])
                image_tile_summary('training/reconstructions',decoded, rows=4, cols=4, shape=params['data_shape'])
            else:
                image_tile_summary('test/inputs',features, rows=4, cols=4, shape=params['data_shape'])
                image_tile_summary('test/reconstructions',decoded, rows=4, cols=4, shape=params['data_shape'])
 
            samples  = decoder({'z':prior.sample(params['batch_size'])},as_dict=True)['x']
            image_tile_summary('prior_samples',samples, rows=4, cols=4, shape=params['data_shape'])  

        eval_metric_ops={
                'log_likelihood_at_MAP': tf.compat.v1.metrics.mean(map_likelihood),
                'log_prior_at_MAP': tf.compat.v1.metrics.mean(map_prior),
                'KL_divergence': tf.compat.v1.metrics.mean(kl),
                'sigma':tf.compat.v1.metrics.mean(sigma_reg),
                'elbo':tf.compat.v1.metrics.mean(objective_VAE),
        #        'KL_laplace_prior':tf.metrics.mean(kl_laplace_prior),
        #        'KL_laplace_vmf':tf.metrics.mean(kl_laplace_vmf),
        #        'Gauss_Newton_determinant': tf.metrics.mean(det),
                #'full_Hessian_determinant': tf.metrics.mean(detC),
        }

        eval_summary_hook = tf.estimator.SummarySaverHook(save_steps=1,output_dir=params['model_dir'],summary_op=tf.compat.v1.summary.merge_all())
        evaluation_hooks  = [eval_summary_hook]
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops = eval_metric_ops, evaluation_hooks=evaluation_hooks)
    else:
        predictions = {'lnp(MAP|x)':tf.reduce_mean(input_tensor=map_prior+map_likelihood)}
    
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
