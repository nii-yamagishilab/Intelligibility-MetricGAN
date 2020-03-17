# Speech Intelligibility Enhancement using GAN 	<br/>(iMetricGAN, PyTorch implementation)


## Usage steps


### 1. Install dependencies

* Install SIIB measure [PySIIB](https://github.com/kamo-naoyuki/pySIIB) (python version)

* Another Dependencies:
    * python 3.7
    * librosa==0.7.1
    * numpy==1.17.2
    * scipy==1.3.1
    * torch==1.2.0
    * tqdm==4.36.1
    * matplotlib==3.1.1

### 2. Prepare training data

Prepare your training data and change data path in **MultiGAN.py**

A toy dataset format example is given in ***./database***

### 3. Training

run: `python MultiGAN.py`

Training configurations can be modified according to your need, `e.g. GAN_epoch, num_of_sampling`

models will be saved in ***./chkpt*** 

### 4. Inference

Prepare the test data, then change paths in **inference.py**

run: `python inference.py`

A pre-trained model is provided in  ***./trained_model***  <br/>It was trained using 44.1 kHz speech materials at RMS=0.02. So please normalize your 44.1kHz raw speech input to RMS=0.02 if you would like to use this pre-trained model.

---
Note: This project was partially based on [MetricGAN](https://github.com/JasonSWFu/MetricGAN) codes.
