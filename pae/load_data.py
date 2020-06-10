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

# functions to load different datasets (mnist, cifar10, random Gaussian data) 

import gzip, zipfile, tarfile
import os, shutil, re, string, urllib, fnmatch
import pickle as pkl
import numpy as np
import sys
import imageio
from PIL import Image
        
def _download_mnist(dataset):
    """
    download mnist dataset if not present
    """
    origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
    print('Downloading data from %s' %origin)
    urllib.request.urlretrieve(origin, dataset)


def _download_cifar10(dataset):
    """
    download cifar10 dataset if not present
    """
    origin = ('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    
    print('Downloading data from %s' % origin)
    urllib.request.urlretrieve(origin,dataset)

def _download_fmnist(dataset,subset,labels=False):

    if subset=='test':
        subset = 't10k'
    if labels:
        origin = ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/%s-labels-idx1-ubyte.gz'%subset)
    else:
        origin = ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/%s-images-idx3-ubyte.gz'%subset)

    print('Downloading data from %s' % origin)
    urllib.request.urlretrieve(origin,dataset)


def _get_datafolder_path():
    """
    returns data path
    """
    #where am I? return full path
    full_path = os.path.abspath('./')
    path = full_path +'/data'
    return path


def load_banana(data_dir,flatten=True,params=None):

    dataset = os.path.join(data_dir,'banana/60000_randomseed111.pkl')

    train_set, valid_set, test_set, _, _,_ = pkl.load(open(dataset,'rb'))

    train_set = train_set.T
    valid_set = valid_set.T
    test_set  = test_set.T

    return train_set, np.arange(len(train_set)), valid_set, np.arange(len(valid_set)), test_set, np.arange(len(test_set))

def load_mnist(data_dir,flatten=True,params=None):
    """
    load mnist dataset
    """

    dataset=os.path.join(data_dir,'mnist/mnist.pkl.gz')

    if not os.path.isfile(dataset):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            print('creating ', datasetfolder)
            os.makedirs(datasetfolder)
        _download_mnist(dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pkl.load(f, encoding='latin1')
    f.close()

    if flatten:
        x_train, targets_train = train_set[0], train_set[1]
        x_test,  targets_test  = test_set[0], test_set[1]
        x_valid, targets_valid = valid_set[0], valid_set[1]
    else:
        x_train, targets_train = train_set[0].reshape((-1,28,28,1)), train_set[1]
        x_test,  targets_test  = test_set[0].reshape((-1,28,28,1)), test_set[1]
        x_valid, targets_valid = valid_set[0].reshape((-1,28,28,1)), valid_set[1]
    
    return x_train*255., targets_train, x_valid*255., targets_valid, x_test*255., targets_test


def load_fmnist(data_dir,flatten=True,params=None):
   
    data = {}
    for subset in ['train','test']:
        data[subset]={}
        for labels in [True,False]:
            if labels:
                dataset=os.path.join(data_dir,'fmnist/fmnist_%s_labels.gz'%subset)
            else:
                dataset=os.path.join(data_dir,'fmnist/fmnist_%s_images.gz'%subset)
            datasetfolder = os.path.dirname(dataset)
            if not os.path.isfile(dataset):
                if not os.path.exists(datasetfolder):
                    os.makedirs(datasetfolder)
                _download_fmnist(dataset,subset,labels)
            with gzip.open(dataset, 'rb') as path:
                if labels:
                    data[subset]['labels'] = np.frombuffer(path.read(), dtype=np.uint8,offset=8)
                else:
                    data[subset]['images'] = np.frombuffer(path.read(), dtype=np.uint8,offset=16)
                    
    x_train = data['train']['images'].reshape((-1,28*28))
    x_test  = data['test']['images'].reshape((-1,28*28))
    if not flatten:
        x_train = x_train.reshape((-1,28,28,1))
        x_test  = x_test.reshape((-1,28,28,1))

    y_train = data['train']['labels']
    y_test  = data['test']['labels']

    return x_train, y_train, x_test, y_test, None, None


def reshape_cifar(x,flatten):
    x = x.reshape([-1, 3, 32, 32])
    x = x.transpose([0, 2, 3, 1])
    if flatten:
        x.reshape(-1,3*32*32)
    return x

def load_cifar10(data_dir,flatten=True,params=None):
    """   
    load cifar10 dataset
    """

    dataset = os.path.join(data_dir,'cifar10/cifar-10-python.tar.gz')

    datasetfolder = os.path.dirname(dataset)
    if not os.path.isfile(dataset):
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_cifar10(dataset)
        with tarfile.open(dataset) as tar:
            tar.extractall(path=datasetfolder)
        
    for i in range(5):
        batchName = os.path.join(datasetfolder,'cifar-10-batches-py/data_batch_{0}'.format(i + 1))
        with open(batchName, 'rb') as f:
            d = pkl.load(f, encoding='latin1')
            data = d['data']
            label= d['labels']
            try:
                train_x = np.vstack((train_x,data))
            except:
                train_x = data
            try:
                train_y = np.append(train_y,np.asarray(label))
            except:
                train_y = np.asarray(label)
                
    batchName = os.path.join(datasetfolder,'cifar-10-batches-py/test_batch')
    with open(batchName, 'rb') as f:
        d = pkl.load(f, encoding='latin1')
        data = d['data']
        label= d['labels']
        test_x = data
        test_y = np.asarray(label)

    train_x = reshape_cifar(train_x,flatten)
    test_x  = reshape_cifar(test_x,flatten)

    print(np.amax(test_x))
    return train_x, train_y, test_x, test_y, None, None


def load_celeba_data(data_dir, flag='training', side_length=None, num=None):
    dir_path = os.path.join(data_dir, 'celeba/img_align_celeba')
    filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
    assert len(filelist) == 202599
    if flag == 'training':
        start_idx, end_idx = 0, 162770
    elif flag == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599

    imgs = []
    for i in range(start_idx, end_idx):
        img = np.array(imageio.imread(dir_path + os.sep + filelist[i]))
        img = img[45:173,25:153]
        if side_length is not None:
            img = Image.fromarray(img)
            img = np.asarray(img.resize([side_length, side_length]))
        new_side_length = np.shape(img)[1]
        img = np.reshape(img, [1, new_side_length, new_side_length, 3])
        imgs.append(img)
        if num is not None and len(imgs) >= num:
            break
        if len(imgs) % 5000 == 0:
            print('Processing {} images...'.format(len(imgs)))
    imgs = np.concatenate(imgs, 0)

    return imgs.astype(np.uint8)


def load_celeba(data_dir, flatten=False, params=None):
    dimensions = params['celeba_dim']
    try:
        x_val   = np.load(os.path.join(data_dir, 'celeba%d'%dimensions,'val.npy'))
    except:
        if not os.path.isdir(os.path.join(data_dir, 'celeba%d'%dimensions)):    
            os.makedirs(os.path.join(data_dir, 'celeba%d'%dimensions))
        print('loading validation data')
        x_val   = load_celeba_data(data_dir, 'val', dimensions)
        np.save(os.path.join(data_dir, 'celeba%d'%dimensions, 'val.npy'), x_val)
    try:
        x_test  = np.load(os.path.join(data_dir, 'celeba%d'%dimensions,'test.npy'))
    except:
        print('loading test data')
        x_test  = load_celeba_data(data_dir,'test', dimensions)
        np.save(os.path.join(data_dir, 'celeba%d'%dimensions, 'test.npy'), x_test)
    try:
        x_train = np.load(os.path.join(data_dir, 'celeba%d'%dimensions,'train.npy'))
    except:
        print('loading training data')
        x_train = load_celeba_data(data_dir,'training', dimensions)
        np.save(os.path.join(data_dir, 'celeba%d'%dimensions, 'train.npy'), x_train)

    return  x_train, None, x_test, None, x_val, None
