# Speech Intelligibility Enhancement using GAN 	

Implementation of the paper: [iMetricGAN: Intelligibility Enhancement for Speech-in-Noise usingGenerative Adversarial Network-based Metric Learning)]()

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

### Authors
* Haoyu Li
* Szu-Wei Fu
* Yu Tsao
* [Junichi Yamagishi](https://nii-yamagishilab.github.io/)

### Acknowlegment

This work was partially supported by a JST CREST Grant (JPMJCR18A6, VoicePersonae project), Japan, and by MEXT KAKENHI Grants (16H06302, 17H04687, 18H04120, 18H04112, 18KT0051, 19K24372), Japan. The numerical calculations were carried out on the TSUBAME 3.0 supercomputer at the Tokyo Institute of Technology.


This project was partially based on [MetricGAN](https://github.com/JasonSWFu/MetricGAN) codes.
