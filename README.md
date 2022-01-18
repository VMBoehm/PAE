# Probabilistic Auto-Encoder (PAE)

***NEW***: Fully modular, easy-to-use pytorch implementation with tutorial can be found at https://github.com/VMBoehm/PytorchPAE

An easy-to-train and evaluate, reliable generative model that achieves state of the art results in data generation, outlier detection and data inpainting and denoising.

For full details, see:
https://arxiv.org/abs/2006.05479.

**Contributors:** Vanessa Böhm, Uroš Seljak

### _Latest Updates:_
We have added [a new simplified notebook](https://github.com/VMBoehm/PAE/blob/master/TrainNVP_simplified_and_explained.ipynb) with detailed instructions for training the second PAE stage (the normalizing flow). The notebook can be run on Google Colab (status Jan '21). 

## How does it work?
The PAE is a two-stage generative model, composed of an Auto-Encoder (AE) that is interpreted probabilistically _after_ training with a Normalizing Flow (NF). The AE compresses the data and maps it to a lower dimensional latent space. The Normalizing Flow is used as a density estimator in the AE-encoded space. The PAE can be interpreted as non-linear generalization of probabilistic low-rank PCA or a regularized Normalizing Flow. It generalizes the idea of regularization to reduce the effect of undesirable singular latent space variables to non-linear models.

An illustration of the PAE:   
<img src="/figures/for_readme-4a.png" alt="drawing" width="300"/>   
The AE encoder and decoder are marked in gray. The AE is trained to minimize the reconstruction error, that is the difference between input (left side) and reconstructed image (right side). The latent space distribution of the AE (pink) can be very irregular, but it can be mapped to a Gaussian (blue) by means of an NF (arrows).


### Sampling

<img src="/figures/for_readme-4b.png" alt="drawing" width="250"/>

Samples are drawn by first sampling from the latent space distribution of the NF (standard normal distribution). The NF maps these samples to the latent space of the AE and the AE decoder maps them into data space. This procedure does not only achieve excellent sample quality by mapping samples to regions of high density in the AE latent space, it also ensures that the entire data space is covered.

Fake Celeb-A images generated with a PAE at latent space dimensionality 64.    
<img src="/figures/for_readme-6.png" alt="drawing" width="250"/> 

Interpolation between data points show that the PAE latent space is continuous.   
<img src="/figures/for_readme-1.png" alt="drawing" width="450"/> 
<img src="/figures/for_readme-3.png" alt="drawing" width="450"/>

### Advantages
The two stage training allows to first reach optimal reconstruction error and then optimal sample quality. Typical VAE training procedures have to balance these two objectives, leading to suboptimal results in both reconstruction and sample quality.
The training of both components of the PAE is simple, stable and does not require hyper-parameter tuning. We provide a python package for training AEs that supports different architectures and data augmentation schemes. Normalizing Flows often struggle with high data dimensionality but are straightforward to apply to the low dimensional AE latent space. We provide notebooks with several different NF implementations that should be sufficient for most AE latent space density estimations. 

### Out-of-Distribution Detection
We find that the log probability in the AE latent space is an excellent outlier-detection metric, outperforming other OoD detectors.

<img src="/figures/for_readme-10.png" alt="drawing" width="600"/> 

### Data Inpainting and Denoising
The PAE can be used for data inputation with uncertainty quantification and we provide notebooks with examples.
High quality reconstructions (left plot, middle coumn) can be obtained from heavily corrupted data (left plot, left column). The underlying true images are shown in the right column. Posterior analysis (central plot) allows for uncertainty quantification from posterior samples (right plot). The first example is compatible with both a 3 and a 5 and the samples reflect this.  
<img src="/figures/for_readme-9.png" alt="drawing" width="600"/> 


## Reproducing PAE results
Our results are obtained in 3 steps  
1) we train Auto-Encoders (AEs) and Variational Auto-Encoders (VAEs)   
2) we train Normalizing Flows (NF) on the encoded data  
3) we analyze the performance of our models   
    a) we measure reconstruction errors and FID scores  
    b) we run Out-of-Distribution detection tests  
    c) we perform data inpainting and denoising with posterior analysis

#### 1) AE/VAE Training
We provide a python package that automizes the AE and (beta-)VAE trainings.   
Running   

```python main.py --helpfull```   

displays all the options for running the code. 
Parameters include the data set, the latent space dimensionality, locations for storing results, the type of loss (AE or VAE), VAE parameters etc.
The model trains by default on Fashion-MNIST, the default parameters are set to reproduce submitted results.   
We further provide trained modules to reproduce our submitted results (download link below).

#### 2) Training the NF
Once the AE is trained, the training of the NFs is performed in jupyter notebooks. We choose this setting, because it allows for easy on the spot visualization and analysis. We provide the original notebooks that were used to train the NF modules in the paper. The trained modules are also inlcuded in the download link.  
-*TrainNVP_fmnist.ipynb*   
-*Train_NSF_FMNIST_64.ipynb*    
-*Train_NSF_FMNIST_128.ipynb*  
-*Train_NSF_celeb_64.ipynb*    

#### 3a) Measurement of Reconstruction Errors and FID scores
The performance of the models in terms of FID scores and reconstruction errors can be analyzed with notebooks named   
-*FIDScore_and_Reconstruction_Error-X.ipynb* (*X*  here is a placeholder)

#### 3b) Out-of-Distribution Detection Tests
The OoD tests can be reproduced with   
-*Out-of-Distribution-Detection.ipynb*   

#### 3c) Posterior Analysis
Image inpainting and denoising is performed in   
-*ImageCorruptionMNIST-solidmask.ipynb*   
-*ImageCorruptionMNIST-noise.ipynb*    

## Trained Models and Parameter Files
The trained models that were used to obtain the submitted results can be obtained from https://zenodo.org/record/3889319 . For anonymization, all paths have been changed to relative paths, expecting that the modules are unzipped in the PAE directory. If done otherwise, the module paths (params['module_path']) has to be adapated.

Each (V)AE module comes with a corresponding parameter file in the `params/` folder. The parameter files contain the complete specification of the setttings under which the module was trained. In most notebooks a simple change of the name of the parameter file will result in the corresponding module being loaded and analyzed. 

## Figure and Table Index
- Fig.2 and Table 1 use modules in `modules/fmnist/class-1/latent_size32/`, and notebooks *FIDScore_and_Reconstruction_Error-X.ipynb*.     
- Fig.3 uses modules in `modules/fmnist/class-1/latent_size64/` and `modules/fmnist/class-1/latent_size128/`.    
- Table 2 can be reproduced with *Out-of-Distribution-Detection.ipynb* and modules in `modules/fmnist/class-1/latent_size64/`  
- Fig. 4 uses modules in `modules/celeba/class-1/latent_size64/` and can be reproduced with *FIDScore_and_Reconstruction_Error-celeba.ipynb* 
- Fig. 6 can be reproduced with modules in `modules/mnist/class-1/latent_size8/` and the notebooks *ImageCorruptionMNIST-solidmask.ipynb* and *ImageCorruptionMNIST-noise.ipynb*


## Setup and Dependencies

The PAE package can be installed from inside the PAE directory with  
``` pip install -e .```

The python package requires tensorflow<=1.15 and compatible releases of tensorflow-hub and tensorflow-probability.

Some of the notebooks require tensorflow2, specifically  
tensorflow-gpu            2.2.0                   
tensorflow-hub            0.8.0                    
tensorflow-probability    0.10.0

The OoD detection requires    
tensorflow-datasets       3.1.0

and the FID scores require pytorch    
torch                     1.4.0

Running the PAE on celeba will require downloading the celeba dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and preprocessing it with the **`load_celeba_data`** function in **`pae/load_data.py`**
