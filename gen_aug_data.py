#coding=utf-8

import glob
import librosa
import os
import numpy as np

savepath = '/home/smg/haoyuli/datasets/clean_SPN'
filelists = glob.glob(os.path.join('/Database/LISTA_Free-Corpus/Sharvard/sharvard_male','*.wav'))
np.random.shuffle(filelists)
fs = 44100

for i in range(710):
    wav, sr = librosa.load(filelists[i], sr=fs)
    #margin = int(fs * 0.1)
    margin = 1
    wav = wav[margin:-margin]
    wav = librosa.effects.trim(
        wav, top_db=30, frame_length=int(fs*0.025), hop_length=int(fs*0.01))[0]
    assert sr==fs
    savename = os.path.join(savepath,'cleanSPN_'+str(1+i)+'.wav')
    librosa.output.write_wav(savename,wav,fs)
