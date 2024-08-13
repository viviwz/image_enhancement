# Image Enhancement using pytorch-3dunet

## Install environment
Create conda environment from environment.yaml by following command: conda env create -f environment.yml. You can change the environment name in the yml file

Activate environment conda activate 3dunet (change 3dunet to the name in .yml file)

(Alternative) If above doesn't work, try to follow the pytorch-3dunet instruction

Install pytorch-3dunet, following instructions from https://github.com/wolny/pytorch-3dunet.git
Install ipywidgets, matplotlib, imageio manually

## Train the model
Use Image_Enhancement_with_3dUnet.ipynb to train your model step by step.

## Use the model
Use Image_Enhancement_3DUnet_Prediction.ipynb to use a trained model for detection.
