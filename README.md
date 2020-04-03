# Speech Intelligibility Enhancement using GAN 	

Implementation of the paper: [iMetricGAN: Intelligibility Enhancement for Speech-in-Noise usingGenerative Adversarial Network-based Metric Learning](https://arxiv.org/abs/2004.00932)

[Audio samples](https://nii-yamagishilab.github.io/samples-iMetricGAN)

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


---
### License

BSD 3-Clause License

Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
