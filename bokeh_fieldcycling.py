import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import stelardatafile as sdf
from bokeh.charts import Scatter, output_file, show
from bokeh.sampledata.autompg import autompg as df

from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure


polymer=sdf.StelarDataFile('297K.sdf',r'.\data')
polymer.sdfimport()

### preprocessing of the data experiment number ie=1
ie=1
parameters=polymer.getparameter(ie)
bs=int(parameters['BS'])
try:
    nblk=int(parameters['NBLK'])
except:
    nblk=1;
ns=int(parameters['NS'])
try:
    dw=parameters['DW']*1e-6 #dwell time is in [mu s]
except:
    dw=1
temperature=parameters['TEMP']
fid=pd.DataFrame(polymer.getdata(ie),index=np.linspace(dw,dw*bs*nblk,bs*nblk),
                 columns=['real', 'im'])/ns
fid['abso']=( fid['real']**2 + fid['im']**2 )**0.5 # last two lines may represent a seperate method

tau=np.logspace(-3,np.log10(5*parameters['T1MX']),nblk) #as a dummy
startpoint=int(0.05*bs)-1
endpoint=int(0.1*bs)
phi=np.zeros(nblk)
for blk in range(nblk):
    start=startpoint + blk * bs
    end=endpoint + blk * bs
    phi[blk]=fid['abso'].iloc[start:end].sum() / (endpoint-startpoint)
df = pd.DataFrame(data=np.c_[tau, phi], columns=['tau', 'phi'])

p1 = figure(plot_width=300, plot_height=300)
p1.multi_line(xs=[fid.index, fid.index, fid.index],
              ys=[fid.im, fid.real, fid.abso],
              color=['blue', 'green', 'red'])
tab1 = Panel(child=p1, title="raw data")

p2 = figure(plot_width=300, plot_height=300)
p2.circle_cross(x=np.array(df.tau), y=np.array(df.phi), color="navy")
tab2 = Panel(child=p2, title="exp decay")

output_file("slider.html")

tabs = Tabs(tabs=[ tab1, tab2 ])

show(tabs)
