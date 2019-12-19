# coding=utf-8

from joblib import Parallel, delayed
import shutil
import scipy.io
import librosa
import os
import time  
import numpy as np
import numpy.matlib
import random
import subprocess

import torch
import torch.nn as nn
from audio_util import *
from pystoi.stoi import stoi
from model import Generator, Discriminator
from dataloader import *
from tqdm import tqdm

pt_dir = './chkpt'
test_dir = './tmp/'

batch_size=1
fs = 44100
num_of_valid_sample=810
chkpt_path = '/home/smg/haoyuli/Project-SI/MultiGAN/chkpt/'


print('Reading path of validation data...')
Test_Noise_path ='/home/smg/haoyuli/SiibGAN/database/Test/Noise/'
Test_Clean_path = '/home/smg/haoyuli/SiibGAN/database/Test/Clean/'
Generator_Test_paths = get_filepaths('/home/smg/haoyuli/SiibGAN/database/Test/Clean/') 

G = Generator().cuda()
Test_STOI = []
Test_SIIB = []


init_epoch = 59

for tepoch in range(5):
    creatdir(test_dir)
    epoch = init_epoch+tepoch
    chkpt = os.path.join(chkpt_path,'chkpt_'+str(epoch)+'.pt')
    G.load_state_dict(torch.load(chkpt)['enhance-model'])
    G.eval()
    Test_enhanced_Name = []
    with torch.no_grad():
        for i, path in enumerate(Generator_Test_paths[0:num_of_valid_sample]):
            S = path.split('/')
            wave_name = S[-1]

            clean_wav,_ = librosa.load(path, sr=fs)
            noise_wav,_ = librosa.load(Test_Noise_path+wave_name, sr=fs)
            noise_mag,noise_phase = Sp_and_phase(noise_wav, Normalization=True)
            clean_mag,clean_phase = Sp_and_phase(clean_wav, Normalization=True)
            
            clean_in = clean_mag.reshape(1,clean_mag.shape[0],-1)
            clean_in = torch.from_numpy(clean_in).cuda()
            noise_in = noise_mag.reshape(1,noise_mag.shape[0],-1)
            noise_in = torch.from_numpy(noise_in).cuda()

            mask = G(clean_in, noise_in)
            clean_power = torch.pow(clean_in, 2/0.30)
            beta_2 = torch.sum(clean_power) / torch.sum(torch.pow(mask,2)*clean_power)
            beta_p = beta_2 ** (0.30/2)
            enh_mag = clean_in * torch.pow(mask, 0.30) * beta_p
            enh_mag = (enh_mag**(1/0.30)).detach().cpu().squeeze(0).numpy()
            enh_wav = SP_to_wav(enh_mag.T, clean_phase)

            enhanced_name = os.path.join(test_dir,wave_name[0:-4]+"@1"+wave_name[-4:])
            librosa.output.write_wav(enhanced_name, enh_wav, fs)
            Test_enhanced_Name.append(enhanced_name) 
    
    # Calculate True STOI
    test_STOI = read_batch_STOI(Test_Clean_path, Test_Noise_path, Test_enhanced_Name)
    Test_STOI.append(np.mean(test_STOI))
    # Calculate True SIIB
    test_SIIB = read_batch_SIIB(Test_Clean_path, Test_Noise_path, Test_enhanced_Name)
    Test_SIIB.append(np.mean(test_SIIB))

    print('%d finished'%epoch)
    print('ESTOI: Modified speech %.4f'%(Test_STOI[tepoch]))
    print('SIIB:  Modified speech %.4f'%(Test_SIIB[tepoch]))
    print('----------------')
    shutil.rmtree(test_dir)

print('All finieshed')
print(Test_STOI)
print(Test_SIIB)