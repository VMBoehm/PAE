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
We further provide trained modules to reproduce our submitted results (download link below).

#### 2) Training the NF
Once the AE is trained, the training of the NFs is performed in jupyter notebooks. We choose this setting, because it allows for easy on the spot visualization and analysis. We provide the original notebooks that were used to train the NF modules in the paper. The trained modules are also inlcuded in the download link.  
-*TrainNVP_fmnist.ipynb*   
-*Train_NSF_FMNIST_64.ipynb*    
-*Train_NSF_FMNIST_128.ipynb*  
-*Train_NSF_celeb_64.ipynb*    

#### 3a) Measurement of Reconstruction Errors and FID scores
The performance of the models in terms of FID scores and reconstruction errors is analyzed with notebooks named   
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
- Fig.3 uses modules in `modules/fmnist/class-1/latent_size64/` and `modules/fmnist/class-1/latent_size128/`. The PAE samples can be visualized in the training notebooks *Train_NSF_FMNIST_64.ipynb* and  *Train_NSF_FMNIST_128.ipynb* or with one of the FID score notebooks.
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
