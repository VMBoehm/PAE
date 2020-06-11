# Probabilistic Auto-Encoder (PAE)

Anonymized code, ipython notebooks, trained modules and instructions for reproducing PAE results.

## Reproducing results
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
We further provide trained modules to reproduce results (download link below).

#### 2) Training the NF
Once the AE is trained, the training of the NFs is performed in jupyter notebooks. We choose this setting, because it allows for easy on the spot visualization and analysis. We provide the original notebooks that were used to train the NF modules that are also inlcuded in the download link.  
*TrainNVP_fmnist.ipynb*
*Train_NSF_FMNIST_64.ipynb*    
*Train_NSF_FMNIST_128.ipynb*  
*Train_NSF_celeb_64.ipynb*    

#### 3a) Measurement of Reconstruction Errors and FID scores
The performance of the models in terms of FID scores and reconstruction errors was analyzed with notebooks named
*FIDScore_and_Reconstruction_Error-X.ipynb*  

#### 3b) Out-of-Distribution Detection Tests
The OoD tests can be reproduced with
*Out-of-Distribution-Detection.ipynb*   

#### 3c) Posterior Analysis
Image inpainting and denoising is performed in
*ImageCorruptionMNIST-solidmask.ipynb*   
and
*ImageCorruptionMNIST-noise.ipynb*   

## Trained Models
The trained models that were used to obtain the submitted results can be obtained from. For anonymization, all paths have been changed to relative paths, expecting that the modules are unzipped in the PAE directory. If done otherwise, the module paths (params['module_path']) will have to be adapated.

## Figure Index



## Setup and Dependencies


The package itself can be installed with   
``` pip install -e .``` (from inside the PAE directory)

The python package requires tensorflow<=1.15 and compatible releases of tensorflow-hub and tensorflow-probability.

Some of the notebooks require tensorflow2, specifically  
tensorflow-gpu            2.2.0                   
tensorflow-hub            0.8.0                    
tensorflow-probability    0.10.0




<sup>[1]</sup> : running the PAE on celeba will require downloading the celeba dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and preprocessing it with the **`load_celeba_data`** function in **`pae/load_data.py`**
