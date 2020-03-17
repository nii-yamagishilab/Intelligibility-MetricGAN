#coding=utf-8
import librosa
import numpy as np
import scipy
from pysiib import SIIB
from pystoi.stoi import stoi
import os 
from joblib import Parallel, delayed

gmtband = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 19, 21, 23, 26, 29, 32, 35, 39, 43, 47, 51, 56, 62, 68, 74, 81, 89, 98, 106, 116, 127, 139, 152, 166, 181, 197]
NB_BANDS = 40
win_hamm = None

fs = 44100

def compute_band_E(x):
    # X: magnitude of spectrogram (T x 513)
    X = np.abs(STFT(x)).T
    global gmtband
    global NB_BANDS
    T, D = X.shape[0],X.shape[1]
    OUT = np.zeros((T,NB_BANDS),dtype=np.float32)

    for iT in range(T):
        sumE = np.zeros(NB_BANDS)

        for i in range(NB_BANDS-1):
            band_size = gmtband[i+1] - gmtband[i]

            for j in range(band_size):
                frac = float(j) / band_size
                tmp = X[iT,gmtband[i]+j]**2
                sumE[i] += (1-frac) * tmp
                sumE[i+1] += frac * tmp


        OUT[iT,:]=sumE
    return OUT


def build_win_hamm(nw,nm):
    win = scipy.hamming(nw)
    win = win/(0.54*nw/nm)  
    return win

def STFT(x, nw=1024,nm=512):
    global win_hamm
    if win_hamm is None:
        win_hamm = build_win_hamm(nw,nm)
    X = librosa.stft(x, nw, nm, nw, window=win_hamm, center=False)
    return X

def ISTFT(X,nw=1024,nm=512):
    global win_hamm
    if win_hamm is None:
        win_hamm = build_win_hamm(nw,nm)
    x = librosa.istft(X,nm,nw,window=win_hamm,center=False)
    return x

def Resyn(mag, phase, alpha):
    # mag,phase 513xT
    # alpfa Tx40  amplifier factor for energy
    T = alpha.shape[0]
    gain = np.zeros([513,T])
    alpha = np.sqrt(alpha)

    for t in range(T):
        g = interp_band_gain(alpha[t,:])
        gain[:,t] = g

    mag = gain * mag
    return SP_to_wav(mag, phase)

def interp_band_gain(bandE):
    global gmtband
    global NB_BANDS
    FREQ_SIZE = 513
    
    g = np.ones(FREQ_SIZE)

    for i in range(NB_BANDS-1):
        band_size = gmtband[i+1] - gmtband[i]
        for j in range(band_size):
            frac = float(j) / band_size
            g[gmtband[i]+j] = (1-frac)*bandE[i] + frac * bandE[i+1]
    return g


def SIIB_Wrapper_eng(x,y,fs):
    minL = min(len(x),len(y))
    x = x[:minL]
    y = y[:minL]
    M = len(x)/fs
    if(M<20):
        x = np.hstack([x]*round(50/M))
        y = np.hstack([y]*round(50/M))
    
    return mapping_func_eng(SIIB(x,y,fs,gauss=True))

def mapping_func_eng(x):
    y = 1/(1+np.exp(-0.05*(x-35)))
    return y


def SIIB_Wrapper_ger(x,y,fs):
    minL = min(len(x),len(y))
    x = x[:minL]
    y = y[:minL]
    M = len(x)/fs
    if(M<20):
        x = np.hstack([x]*round(50/M))
        y = np.hstack([y]*round(50/M))
    
    return mapping_func_ger(SIIB(x,y,fs,gauss=True))

def mapping_func_ger(x):
    y = 1/(1+np.exp(-0.09*(x-25)))
    return y



def read_STOI(clean_root, noise_root, enhanced_file):
    f=enhanced_file.split('/')[-1]
    wave_name=f.split('_')[-1].split('@')[0]
    
    clean_wav,_    = librosa.load(clean_root+'Train_'+wave_name+'.wav', sr=fs)
    noise_wav,_    = librosa.load(noise_root+'Train_'+wave_name+'.wav', sr=fs)     
    enhanced_wav,_ = librosa.load(enhanced_file, sr=fs)
    minL = min(len(clean_wav),len(enhanced_wav))
    clean_wav = clean_wav[:minL]
    noise_wav = noise_wav[:minL]
    enhanced_wav = enhanced_wav[:minL]
    stoi_score = stoi(clean_wav, enhanced_wav + noise_wav, fs, extended=True) * 2    
    return stoi_score
    
# Parallel computing for accelerating    
def read_batch_STOI(clean_root, noise_root, enhanced_list):
    stoi_score = Parallel(n_jobs=30)(delayed(read_STOI)(clean_root, noise_root, en) for en in enhanced_list)
    return stoi_score

def read_SIIB(clean_root, noise_root, enhanced_file):
    f=enhanced_file.split('/')[-1]
    wave_name=f.split('_')[-1].split('@')[0]
    
    clean_wav,_    = librosa.load(clean_root+'Train_'+wave_name+'.wav', sr=fs)     
    enhanced_wav,_ = librosa.load(enhanced_file, sr=fs)
    noise_wav,_    = librosa.load(noise_root+'Train_'+wave_name+'.wav', sr=fs)     
    minL = min(len(clean_wav),len(enhanced_wav))
    clean_wav = clean_wav[:minL]
    noise_wav = noise_wav[:minL]
    enhanced_wav = enhanced_wav[:minL]
    
    siib_score = SIIB_Wrapper_ger(clean_wav, enhanced_wav + noise_wav, fs)  
    return siib_score
    
# Parallel computing for accelerating    
def read_batch_SIIB(clean_root, noise_root, enhanced_list):
    siib_score = Parallel(n_jobs=30)(delayed(read_SIIB)(clean_root, noise_root, en) for en in enhanced_list)
    return siib_score

def read_SIIB_DRC(clean_root, noise_root, enhanced_file):
    f=enhanced_file.split('/')[-1]
    wave_name = f
    clean_wav,_    = librosa.load(clean_root+wave_name, sr=fs)     
    enhanced_wav,_ = librosa.load(enhanced_file, sr=fs)
    noise_wav,_    = librosa.load(noise_root+wave_name, sr=fs)     
    minL = min(len(clean_wav),len(enhanced_wav))
    clean_wav = clean_wav[:minL]
    noise_wav = noise_wav[:minL]
    enhanced_wav = enhanced_wav[:minL]
    
    siib_score = SIIB_Wrapper_ger(clean_wav, enhanced_wav + noise_wav, fs)  
    return siib_score

def read_batch_SIIB_DRC(clean_root, noise_root, enhanced_list):
    siib_score = Parallel(n_jobs=30)(delayed(read_SIIB_DRC)(clean_root, noise_root, en) for en in enhanced_list)
    return siib_score

def read_STOI_DRC(clean_root, noise_root, enhanced_file):
    f=enhanced_file.split('/')[-1]
    wave_name = f
    clean_wav,_    = librosa.load(clean_root+wave_name, sr=fs)     
    enhanced_wav,_ = librosa.load(enhanced_file, sr=fs)
    noise_wav,_    = librosa.load(noise_root+wave_name, sr=fs)     
    minL = min(len(clean_wav),len(enhanced_wav))
    clean_wav = clean_wav[:minL]
    noise_wav = noise_wav[:minL]
    enhanced_wav = enhanced_wav[:minL]
    
    stoi_score = stoi(clean_wav, enhanced_wav + noise_wav, fs, extended=True) * 2
    return stoi_score

def read_batch_STOI_DRC(clean_root, noise_root, enhanced_list):
    stoi_score = Parallel(n_jobs=30)(delayed(read_STOI_DRC)(clean_root, noise_root, en) for en in enhanced_list)
    return stoi_score

def Corresponding_clean_list(file_list,train_clean_path):
    index=0
    co_clean_list=[]
    while index<len(file_list):
        f=file_list[index].split('/')[-1]
               
        wave_name=f.split('_')[-1]
        clean_name='Train_'+wave_name
            
        co_clean_list.append('1.00,'+train_Clean_path+clean_name)
        index += 1  
    return co_clean_list

def List_concat(score, enhanced_list):
    concat_list=[]
    for i in range(len(score)):
        concat_list.append(str(score[i])+','+enhanced_list[i]) 
    return concat_list

def List_concat_score(score, score2):
    concat_list=[]
    for i in range(len(score)):
        concat_list.append(str(score[i])+','+str(score2[i])) 
    return concat_list

def creatdir(directory):    
    if not os.path.exists(directory):
        os.makedirs(directory) 

def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path
     
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            if '.wav' in filename:
                # Join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


def Sp_and_phase(signal, Normalization=False):        
    #signal_length = signal.shape[0]
    n_fft = 1024
    #y_pad = librosa.util.fix_length(signal, signal_length + n_fft // 2)
    
    F = STFT(signal)
    
    Lp=np.abs(F)
    phase=np.angle(F)
    if Normalization==True:    
        NLp = Lp**0.30
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(NLp.shape[1],513)) # For LSTM
    return NLp, phase #, signal_length

def SP_to_wav(mag, phase, signal_length=None):
    Rec = np.multiply(mag , np.exp(1j*phase))
    result = ISTFT(Rec)
    return result  

