#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import re
from bokeh.charts import Scatter, output_file, show
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.palettes import Spectral5
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Dropdown, Select
from bokeh.models.callbacks import CustomJS
from tornado.ioloop import IOLoop
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider
from bokeh.server.server import Server
import stelardatafile as sdf
from utils import get_x_axis, model_exp_dec, fun_exp_dec, get_mag_amplitude, magnetization_fit
from scipy.optimize import leastsq

#specify and import data file
path=os.path.join(os.path.curdir,'data')
polymer=sdf.StelarDataFile('297K.sdf',path)
polymer.sdfimport()
nr_experiments = polymer.get_number_of_experiments()

# parameters to dataframe
par_df=pd.DataFrame()
for ppiepigpii in range(1):
    par=[]
    for ie in range(nr_experiments):
        par.append(polymer.getparameter(ie+1))
    par_df=pd.DataFrame(par)

# categorize the data
columns=sorted(par_df.columns)
discrete = [x for x in columns if (par_df[x].dtype == object or par_df[x].dtype == str)]
continuous = [x for x in columns if x not in discrete]
time = [x for x in continuous if x=='TIME']
quantileable = [x for x in continuous if len(par_df[x].unique()) > 20]


io_loop = IOLoop.current()
#initially set experiment number ie=1
ie = 1

def modify_doc(doc):
    fid=polymer.getfid(ie) #dataframe
    # TODO: more testing on get_x_axis
    tau = get_x_axis(polymer.getparameter(ie))

    #calculate magnetization:
    startpoint=int(0.05*polymer.getparvalue(ie,'BS'))
    endpoint=int(0.1*polymer.getparvalue(ie,'BS')) #TODO: make a range slider to get start- and endpoint interactively
    phi = get_mag_amplitude(fid, startpoint, endpoint,
                            polymer.getparvalue(ie,'NBLK'),
                            polymer.getparvalue(ie,'BS'))

    #prepare magnetization decay curve for fit
    df = pd.DataFrame(data=np.c_[tau, phi], columns=['tau', 'phi'])
    df['phi_normalized']=(df['phi'] - df['phi'].iloc[0] ) / (df['phi'].iloc[-1] - df['phi'].iloc[1] )
    polymer.addparameter(ie,'df_magnetization',df)
    

    
    fit_option = 2
    p0 = [1 , 2 * polymer.getparvalue(ie,'T1MX')**-1, 0]
    df, popt = magnetization_fit(df, p0, fit_option)
    
    df['fit_phi'] = model_exp_dec(df.tau, *popt)
    

    # convert data to handle in bokeh
    source_fid = ColumnDataSource(data=ColumnDataSource.from_df(fid))
    source_df = ColumnDataSource(data=ColumnDataSource.from_df(df))
    
    # create and plot figures
    p1 = figure(plot_width=300, plot_height=300,
                title='Free Induction Decay', webgl=True)
    p1.line('index', 'im', source=source_fid, color='blue')
    p1.line('index', 'real', source=source_fid, color='green')
    p1.line('index', 'magnitude', source=source_fid, color='red')


    p2 = figure(plot_width=300, plot_height=300,
                title='Magnetization Decay')
    p2.circle_cross('tau', 'phi_normalized', source=source_df, color="navy")
    p2.line('tau', 'fit_phi', source=source_df, color="teal")

    
    # in the plot 4 use followingimpo
    SIZES = list(range(6, 22, 3)) # for some sizes
    COLORS = Spectral5 # for some colors (more colors would be nice somehow)

    def plot_par():
        xs = par_df[x.value].values
        ys = par_df[y.value].values
        x_title = x.value.title()
        y_title = y.value.title()

        kw = dict() #holds optional keyword arguments for figure()
        if x.value in discrete:
            kw['x_range'] = sorted(set(xs))
        if y.value in discrete:
            kw['y_range'] = sorted(set(ys))
        if y.value in time:
            kw['y_axis_type'] = 'datetime'
        if x.value in time:
            kw['x_axis_type'] = 'datetime'
        
            
        kw['title']="%s vs %s" % (x_title, y_title)


        p4 = figure(plot_height=300, plot_width=600, tools='pan,box_zoom,reset',
                    **kw)

        p4.xaxis.axis_label = x_title
        p4.yaxis.axis_label = y_title


        if x.value in discrete:
            p4.xaxis.major_label_orientation = pd.np.pi / 4 # rotates labels...
                    
        sz = 9
        if size.value != 'None':
            groups = pd.qcut(pd.to_numeric(par_df[size.value].values), len(SIZES))
            sz = [SIZES[xx] for xx in groups.codes]

        c = "#31AADE"
        if color.value != 'None':
            groups = pd.qcut(pd.to_numeric(par_df[color.value]).values, len(COLORS))
            c = [COLORS[xx] for xx in groups.codes]
       
        p4.circle(x=xs, y=ys, color=c, size=sz, line_color="white", alpha=0.6, hover_color='white', hover_alpha=0.5)
        return p4

    def update(attr, old, new):
        layout_p4.children[1] = plot_par()

    def cb(attr, old, new):
        ie = source.data['value'][0]
        fid = polymer.getfid(ie)
        source_fid.data = ColumnDataSource.from_df(fid)
        
        try:
            tau = get_x_axis(polymer.getparameter)
            startpoint=int(0.05*polymer.getparvalue(ie,'BS'))
            endpoint=int(0.1*polymer.getparvalue(ie,'BS'))
            phi = get_mag_amplitude(fid, startpoint, endpoint,
                                    polymer.getparvalue(ie,'NBLK'),
                                    polymer.getparvalue(ie,'BS'))
            df = pd.DataFrame(data=np.c_[tau, phi], columns=['tau', 'phi'])
            df['phi_normalized'] = (df['phi'] - df['phi'].iloc[0] ) / (df['phi'].iloc[-1] - df['phi'].iloc[1] )
            polymer.addparameter(ie,'df_magnetization',df)
            fit_option = 2 #mono exponential, 3 parameter fit
            p0=[1.0, polymer.getparvalue(ie,'T1MX')**-1*2, 0]
            df, popt = magnetization_fit(df, p0, fit_option)
            source_df.data = ColumnDataSource.from_df(df)
            polymer.addparameter(ie,'popt(mono_exp)',popt)
        except KeyError:
            print('no relaxation experiment found')
            tau=np.zeros(nblk)
            phi=np.zeros(nblk)
            df = pd.DataFrame(data=np.c_[tau, phi], columns=['tau', 'phi'])
            df['phi_normalized'] = np.zeros(nblk)
            df['fit_phi'] = np.zeros(nblk)
            source_df.data = ColumnDataSource.from_df(df)
        
    #this source is only used to communicate to the actual callback (cb)
    source = ColumnDataSource(data=dict(value=[]))
    source.on_change('data',cb)
    
    slider = Slider(start=1, end=nr_experiments, value=1, step=1,callback_policy='mouseup')
    slider.callback=CustomJS(args=dict(source=source),code="""
        source.data = { value: [cb_obj.value] }
    """)#unfortunately this customjs is needed to throttle the callback in current version of bokeh

    
    # add select boxes for p4
    x = Select(title='X-Axis', value='ZONE', options=columns)
    x.on_change('value', update)

    y = Select(title='Y-Axis', value='TIME', options=columns)
    y.on_change('value', update)

    size = Select(title='Size', value='None', options=['None'] + quantileable)
    size.on_change('value', update)

    color = Select(title='Color', value='None', options=['None'] + quantileable)
    color.on_change('value', update)


    controls_p4 = widgetbox([x,y,color,size], width=150)
    layout_p4 = row(controls_p4,plot_par())

    
    
    #fitting on all experiments
    p3 = figure(plot_width=300, plot_height=300,
            title='normalized phi vs normalized tau', webgl = True,
                y_axis_type = 'log',
                x_axis_type = 'linear')
    
    #fit magnetization decay for all experiments
    r1=np.zeros(nr_experiments)
    for i in range(nr_experiments):
        try:
            par=polymer.getparameter(i)
            fid=polymer.getfid(i)
            tau = get_x_axis(polymer.getparameter(i))
            startpoint=int(0.05*polymer.getparameter(i)['BS'])
            endpoint=int(0.1*polymer.getparameter(i)['BS']) #TODO: make a range slider to get start- and endpoint interactively
            phi = get_mag_amplitude(fid, startpoint, endpoint,
                                    polymer.getparameter(i)['NBLK'], polymer.getparameter(i)['BS'])
            df = pd.DataFrame(data=np.c_[tau, phi], columns=['tau', 'phi'])
            df['phi_normalized']=(df['phi'] - df['phi'].iloc[0] ) / (df['phi'].iloc[-1] - df['phi'].iloc[1] )
            polymer.addparameter(i,'df_magnetization',df)

            p0 = [1, 2 * polymer.getparvalue(i,'T1MX'), 0]
            df, popt = magnetization_fit(df,p0, fit_option=2)
            polymer.addparameter(i,'amp',popt[0])
            polymer.addparameter(i,'r1',popt[1])
            polymer.addparameter(i,'noise',popt[2])
            r1[i] = popt[1]

            tau = popt[1]*df.tau
            phi = popt[0]**-1*(df.phi_normalized - popt[2])
            p3_df=pd.DataFrame(data=np.c_[ tau, phi ], columns=['tau', 'phi'])
            source_p3=ColumnDataSource(data=ColumnDataSource.from_df(p3_df))
            p3.line('tau', 'phi', source=source_p3) #TODO add nice colors
        except KeyError:
            print('no relaxation experiment found')

    par_df['r1']=r1
    doc.add_root(column(slider, p1, p2, p3))
    doc.add_root(layout_p4)
    doc.add_root(source) # i need to add the source for some reason...


bokeh_app = Application(FunctionHandler(modify_doc))

server = Server({'/': bokeh_app}, io_loop=io_loop)
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    io_loop.add_callback(server.show, "/")
io_loop.start()

