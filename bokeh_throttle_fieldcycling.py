#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from bokeh.charts import Scatter, output_file, show
from bokeh.sampledata.autompg import autompg as df
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Dropdown
from bokeh.models.callbacks import CustomJS
from tornado.ioloop import IOLoop
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.server.server import Server
import stelardatafile as sdf
from utils import get_x_axis

#specify and import data file
path=os.path.join(os.path.curdir,'data')
polymer=sdf.StelarDataFile('297K.sdf',path)
polymer.sdfimport()
nr_experiments = polymer.get_number_of_experiments()

io_loop = IOLoop.current()
#initially set experiment number ie=1
ie = 1

def modify_doc(doc):
    #lookup parameters for calculation of fid
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
    #calculate series of fid
    fid=pd.DataFrame(polymer.getdata(ie),index=np.linspace(dw,dw*bs*nblk,bs*nblk),
                     columns=['real', 'im'])/ns
    fid['magnitude']=( fid['real']**2 + fid['im']**2 )**0.5 # last two lines may represent a seperate method
    #calculate the tau for the corresponding fid in each block
    tau=np.logspace(-3,np.log10(5*parameters['T1MX']),nblk) #FIXME we need to read the tau from bini and bend (complicated)
    #calculate magnitization amplitudes from fid series, integrate from startpoint to endpoint
    # TODO: Test the following line:
    # tau = get_x_axis(parameters)
    startpoint=int(0.05*bs)-1
    endpoint=int(0.1*bs)
    phi=np.zeros(nblk)
    for blk in range(nblk):
        start=startpoint + blk * bs
        end=endpoint + blk * bs
        phi[blk]=fid['magnitude'].iloc[start:end].sum() / (endpoint-startpoint)
    df = pd.DataFrame(data=np.c_[tau, phi], columns=['tau', 'phi']) 

    #convert data to handle in bokeh
    source_fid = ColumnDataSource(data=ColumnDataSource.from_df(fid))
    source_df = ColumnDataSource(data=ColumnDataSource.from_df(df))


    #create and plot figures
    p1 = figure(plot_width=300, plot_height=300,
                title='Free Induction Decay', webgl=True)
    p1.line('index', 'im', source=source_fid, color='blue')
    p1.line('index', 'real', source=source_fid, color='green')
    p1.line('index', 'magnitude', source=source_fid, color='red')


    p2 = figure(plot_width=300, plot_height=300,
                title='Magnetization Decay')
    p2.circle_cross('tau', 'phi', source=source_df, color="navy")



    def cb(attr, old, new):
        ie = source.data['value'][0]
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
        fid=pd.DataFrame(polymer.getdata(ie),index=np.linspace(dw,dw*bs*nblk,bs*nblk), columns=['real', 'im'])/ns
        fid['magnitude']=( fid['real']**2 + fid['im']**2 )**0.5 # last two lines may represent a seperate method
        source_fid.data = ColumnDataSource.from_df(fid)
        
        try:
            tau=np.logspace(-3,np.log10(5*parameters['T1MX']),nblk) #as a dummy
            startpoint=int(0.05*bs)-1
            endpoint=int(0.1*bs)
            phi=np.zeros(nblk)  
            for blk in range(nblk):
                start=startpoint + blk * bs
                end=endpoint + blk * bs
                phi[blk]=fid['magnitude'].iloc[start:end].sum() / (endpoint-startpoint)
            df = pd.DataFrame(data=np.c_[tau, phi], columns=['tau', 'phi'])
            source_df.data = ColumnDataSource.from_df(df)
        except KeyError:
            print('no relaxation experiment found')
            tau=np.zeros(nblk)
            phi=np.zeros(nblk)
            df = pd.DataFrame(data=np.c_[tau, phi], columns=['tau', 'phi'])
            source_df.data = ColumnDataSource.from_df(df)
        
    #this source is only used to communicate to the actual callback (cb)
    source = ColumnDataSource(data=dict(value=[]))
    source.on_change('data',cb)
    
    slider = Slider(start=1, end=nr_experiments, value=1, step=1,callback_policy='mouseup')
    slider.callback=CustomJS(args=dict(source=source),code="""
        source.data = { value: [cb_obj.value] }
    """)#unfortunately this customjs is needed to throttle the callback in current version of bokeh


    doc.add_root(column(slider, p1, p2))
    doc.add_root(source) # i need to add the source for some reason...

bokeh_app = Application(FunctionHandler(modify_doc))

server = Server({'/': bokeh_app}, io_loop=io_loop)
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    io_loop.add_callback(server.show, "/")
io_loop.start()

