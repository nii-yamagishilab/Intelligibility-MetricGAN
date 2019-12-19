from audio_util import *
from dataloader import *

noise_path = '/home/smg/haoyuli/SiibGAN/database/Train/Noise/'
clean_path  = '/home/smg/haoyuli/SiibGAN/database/Train/Clean/'
gen_file_list = get_filepaths('/home/smg/haoyuli/SiibGAN/database/Train/Clean/')
genloader = create_dataloader(gen_file_list,noise_path)

x = iter(genloader)

bandClean,bandNoise,clean_mag,clean_phase,noise_mag,noise_phase, target = x.next()


enh_file_list = ['0.2228,/home/smg/haoyuli/SiibGAN/database/Train/Enhance/Train_32.wav']

disloader = create_dataloader(enh_file_list,noise_path,clean_path,'D')
x = iter(disloader)

a,b,c = x.next()

