import librosa
from pystoi.stoi import stoi
from audio_util import *
import numpy as np
import os

enhan_path = '/home/smg/haoyuli/HC/output/Eng_ManyPre/Modify/'
clean_path = '/home/smg/haoyuli/HC/output/Eng_ManyPre/Clean/'
noise_path = '/home/smg/haoyuli/HC/output/Eng_ManyPre/Noise/'

ESTOI_raw = []
ESTOI_mod = []
SIIB_raw = []
SIIB_mod = []

for i in range(810):
    clean, fs = librosa.load(os.path.join(clean_path,'Train_'+str(i+1)+'.wav'),sr=None)
    assert fs==44100
    noise, fs = librosa.load(os.path.join(noise_path,'Train_'+str(i+1)+'.wav'),sr=None)
    enhan, fs = librosa.load(os.path.join(enhan_path,'Train_'+str(i+1)+'.wav'),sr=None)

    ESTOI_raw.append(stoi(clean,clean+noise,fs,extended=True))
    ESTOI_mod.append(stoi(clean,enhan+noise,fs,extended=True))
    SIIB_raw.append(SIIB_Wrapper_eng(clean,clean+noise,fs))
    SIIB_mod.append(SIIB_Wrapper_eng(clean,enhan+noise,fs))
    print("%d finished"%(i+1))

print("result 86:")
print('ESTOI: Raw speech %.4f  Modified speech %.4f'%(np.mean(ESTOI_raw),np.mean(ESTOI_mod)))
print('SIIB: Raw speech %.4f  Modified speech %.4f'%(np.mean(SIIB_raw),np.mean(SIIB_mod)))


