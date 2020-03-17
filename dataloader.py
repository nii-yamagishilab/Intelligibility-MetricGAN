# coding=utf-8

import os

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import scipy.io as scio
from audio_util import *
#import pdb

def toTorch(x):
    return torch.from_numpy(x.astype(np.float32))

class Generator_train_dataset(Dataset):
    def __init__(self, file_list, noise_path):
        self.file_list = file_list
        self.noise_path = noise_path
        self.target_score = np.asarray([1.0, 1.0],dtype=np.float32)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        pmag = 0.30
        pban = 0.20

        file = self.file_list[idx]

        clean_wav,_ = librosa.load(self.file_list[idx], sr=44100)
        noise_wav,_ = librosa.load(self.noise_path+self.file_list[idx].split('/')[-1], sr=44100)

        # already power compression by 0.30
        noise_mag,noise_phase = Sp_and_phase(noise_wav, Normalization=True)
        clean_mag,clean_phase = Sp_and_phase(clean_wav, Normalization=True)

        #bandNoise = compute_band_E(noise_wav) ** pban
        #bandClean = compute_band_E(clean_wav) ** pban

        return clean_mag,clean_phase,noise_mag,noise_phase,self.target_score

class Discriminator_train_dataset(Dataset):
    def __init__(self, file_list, noise_path, clean_path):
        self.file_list = file_list
        self.noise_path = noise_path
        self.clean_path = clean_path
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
        pban = 0.20
        score_filepath = self.file_list[idx].split(',')

        enhance_wav,_ = librosa.load(score_filepath[2], sr=44100)
        enhance_mag, _ = Sp_and_phase(enhance_wav, Normalization=True)
        #pdb.set_trace()
        f = self.file_list[idx].split('/')[-1]
        if '@' in f:
            f = f.split('@')[0] + '.wav'
        noise_wav,_ = librosa.load(self.noise_path+f, sr=44100)
        noise_mag, _ = Sp_and_phase(noise_wav, Normalization=True)

        clean_wav, _ = librosa.load(self.clean_path+f, sr=44100)
        clean_mag, _ = Sp_and_phase(clean_wav, Normalization=True)

        #bandNoise = compute_band_E(noise_wav) ** pban
        #bandEnhan = compute_band_E(enhance_wav) ** pban
        #bandClean = compute_band_E(clean_wav) ** pban

        True_score = np.asarray([float(score_filepath[0]),float(score_filepath[1])],dtype=np.float32)

        #noise_mag, clean_mag, bandNoise, bandClean = noise_mag.T, clean_mag.T, bandNoise.T, bandClean.T
        #enhance_mag, bandEnhan = enhance_mag.T, bandEnhan.T
        noise_mag, clean_mag, enhance_mag = noise_mag.T, clean_mag.T, enhance_mag.T

        noise_mag = noise_mag.reshape(1,513,noise_mag.shape[1])
        clean_mag = clean_mag.reshape(1,513,clean_mag.shape[1])
        enhance_mag = enhance_mag.reshape(1,513,enhance_mag.shape[1])

        #bandNoise = bandNoise.reshape(1,40,bandNoise.shape[1])
        #bandClean = bandClean.reshape(1,40,bandClean.shape[1])
        #bandEnhan = bandEnhan.reshape(1,40,bandEnhan.shape[1])

        return np.concatenate((enhance_mag,noise_mag,clean_mag),axis=0), True_score
    
def create_dataloader(filelist, noise_path, clean_path=None, loader='G'):
    if loader=='G':
        return DataLoader(dataset=Generator_train_dataset(filelist, noise_path),
                          batch_size=1,
                          shuffle=True,
                          num_workers=6,
                          drop_last=True)
    elif loader=='D':
        return DataLoader(dataset=Discriminator_train_dataset(filelist, noise_path, clean_path),
                          batch_size=1,
                          shuffle=True,
                          num_workers=6,
                          drop_last=True)
    else:
        raise Exception("No such dataloader type!")