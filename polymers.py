import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import stelardatafile as sdf

polymer=sdf.StelarDataFile('297K.sdf',r'.\data')
polymer.sdfimport()

#preprocessing of the data experiment number ie=1
ie=1
parameters=polymer.getparameter(ie)
bs=parameters['BS']
try:
    nblk=parameters['NBLK']
except:
    nblk=1;
ns=parameters['NS']
try:
    dw=parameters['DW']*1e-6 #dwell time is in [mu s]
except:
    dw=1
temperature=parameters['TEMP']
fid=pd.DataFrame(polymer.getdata(ie),index=np.linspace(dw,dw*bs*nblk,bs*nblk),
                 columns=['real', 'im'])/ns
fid['abs']=pow(pow(fid['real'],2)+pow(fid['im'],2),0.5)
fid.plot(marker='x',linestyle='none')
plt.show()



