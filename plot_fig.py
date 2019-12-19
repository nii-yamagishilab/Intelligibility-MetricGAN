#coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


data = np.random.randn(8,2)
data = np.array([[0.3051,0.3902],[0.4230,0.5909],[0.4159,0.6227],[0.4734,0.6473],[0.4476,0.7246],[0.3566,0.68],[0.4777,0.6883],[0.4743,0.7018]])
df = pd.DataFrame(abs(data),columns=['ESTOI','SIIB',],index = ['Raw','OptMI','OptSII','SSDRC','SiibG','Z-SiibG','MultiG','M+MultiG'])
fig = df.plot(kind='bar',rot=0, figsize=(8,6),ylim=[0.2,0.8],title='Intelligibility scores with different methods',sort_columns=True)
plt.legend(loc='upper left',fontsize=14)
plt.xticks(size = 12)
plt.savefig('/home/smg/haoyuli/SiibGAN.jpg')
plt.close()


data = np.random.randn(4,3)
data = np.array([[3.6717,11.6211,13.3912],[3.9735,5.8955,6.3971],[3.9157,8.6351,9.8764],[3.9036,9.4721,10.3673]])
df = pd.DataFrame(abs(data),columns=['Noisy','Baseline','NoiseToken'],index = ['Typing','Babble','AirportAnnounce','SqueakyChair'])
fig = df.plot(kind='barh',figsize=(10,8),xlim=[2.5,15],title='SDR with different methods, under unseen noise conditions',sort_columns=True)

plt.savefig('./sdr.jpg')
plt.close()




set_1 = [0,0.2,0.3,0.6,0.9]
set_2 = [0,0.2,0.3,0.4,0.7]

plt.scatter(set_1,set_2,s=10,c="blue")
plt.savefig('/home/smg/haoyuli/test.jpg')
plt.close()
