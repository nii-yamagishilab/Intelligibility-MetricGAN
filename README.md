# Speech Intelligibility Enhancement using GAN 	<br/>(iMetricGAN, PyTorch implementation)

## Usage steps


### 1. Install dependencies

Install SIIB measure (python version) 
[PySIIB](https://github.com/kamo-naoyuki/pySIIB)

* Another Dependencies:
    * python 3.7
    * librosa==0.7.1
    * numpy==1.17.2
    * scipy==1.3.1
    * torch==1.2.0
    * tqdm==4.36.1
    * matplotlib==3.1.1

### 2. Prepare training data

Prepare data and change data path in **MultiGAN.py**

A tiny dataset format example is given in ***./database***

### 3. Training

run: `python MultiGAN.py`

models will be saved in ***./chkpt*** 

### 4. Inference

Prepare the data you would like to process, then change data path and model path in **inference.py**

run: `python inference.py`

A pre-trained model is provided in ***./trained_model*** (It was trained using RMS=0.02 speech materials. So please normalize your raw speech input to RMS=0.02 if you would like to use this pre-trained model.)

---
Note: This project was partially based on [MetricGAN](https://github.com/JasonSWFu/MetricGAN) codes.
