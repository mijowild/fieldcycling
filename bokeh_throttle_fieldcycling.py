#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from bokeh.charts import Scatter, output_file, show
from bokeh.sampledata.autompg import autompg as df
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
from bokeh.models.axes import DatetimeAxis
from bokeh.server.server import Server
import stelardatafile as sdf
from utils import get_x_axis

#specify and import data file
path=os.path.join(os.path.curdir,'data')
polymer=sdf.StelarDataFile('297K.sdf',path)
polymer.sdfimport()
nr_experiments = polymer.get_number_of_experiments()

# parameters to dataframe
par=[]
for ie in range(nr_experiments):
    par.append(polymer.getparameter(ie+1))
par_df=pd.DataFrame(par)

# categorize the data
columns=sorted(par_df.columns)
discrete = [x for x in columns if (par_df[x].dtype == object or par_df[x].dtype == str)]
continuous = [x for x in columns if x not in discrete]
time = [x for x in continuous if x=='TIME']
#continuous.remove('TIME')
quantileable = [x for x in continuous if len(par_df[x].unique()) > 20]


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
    #calculate magnitization amplitudes from fid series, integrate from startpoint to endpoint
    #seems working for the moment # TODO: more testing
    tau = get_x_axis(parameters, nblk)
    startpoint=int(0.05*bs)-1
    endpoint=int(0.1*bs)
    phi=np.zeros(nblk)
    for blk in range(nblk):
        start=startpoint + blk * bs
        end=endpoint + blk * bs
        phi[blk]=fid['magnitude'].iloc[start:end].sum() / (endpoint-startpoint)
    df = pd.DataFrame(data=np.c_[tau, phi], columns=['tau', 'phi'])
    df['phi_normalized']=(df['phi'] - df['phi'].iloc[0] ) / (df['phi'].iloc[-1] - df['phi'].iloc[1] )
    
    # fit exponential decay. 
    # Options:
    #    1) Linearize the system, and fit a line to the log of the data.
    #        - would be prefered, but needs the y-axes offset.
    #    2) Use a non-linear solver (e.g. scipy.optimize.curve_fit

    
    fit_option = 2
    def model_func(t, A, K, C):
        return A * np.exp(- K * t) + C
    def fun(par,t,y):
        A, K, C = par
        return model_func(t, A,K,C) - y
        
    
    if fit_option ==1:
        from utils import fit_exp_linear
        C0 = 0 # offset
        popt = fit_exp_linear(df.tau, df.phi_normalized, C0)
    elif fit_option == 2:
        # needs prior knowledge for p0...
        from scipy.optimize import leastsq
        #popt, _ = curve_fit(model_func, np.array(df.tau), np.array(df.phi_normalized), p0=[1.0, 0.1**-1, 0.0])
        p0=[1.0, 0.1**-1, 0.0]
        popt, _ = leastsq(fun, p0  , args=(np.array(df.tau),np.array(df.phi_normalized)) )
    df['fit_phi'] = model_func(df.tau, *popt)

    

    # get a plot of two parameters:
    # TODO write function which takes parameter name and returns list with values of all values found in the object
    # start with zone, vs time. # TODO choose any parameter from the dropdown and think of nice visualization
##    par_x_name = 'ZONE'
##    par_y_name = 'TIME'
##    # i know there is a pythonic oneliner for the next couple of lines forgive me with lambda: something i dont remember at the moment :)
##    par_y=[]
##    par_x=[]
##    for i in range(nr_experiments):
##        # 
##        
##        try:
##            par_y.append(pd.to_datetime(polymer.getparameter(i)[par_y_name]))
##        except:
##            print('bla')
##            par_y.append(polymer.getparameter(i)[par_y_name])
##        try:
##            par_x.append(pd.to_datetime(polymer.getparameter(i)[par_x_name]))
##        except:
##            print('bla')
##            par_x.append(polymer.getparameter(i)[par_x_name])
##    
##    df_par = pd.DataFrame(data=np.c_[par_x, par_y], columns=[par_x_name, par_y_name])

    # convert data to handle in bokeh
    source_fid = ColumnDataSource(data=ColumnDataSource.from_df(fid))
    source_df = ColumnDataSource(data=ColumnDataSource.from_df(df))
##    source_par = ColumnDataSource(data=ColumnDataSource.from_df(df_par))
    
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


##    p3 = figure(plot_width=300, plot_height=300,
##                title='Parameter Plot',y_axis_type="datetime")
##    p3.circle_cross(par_x_name, par_y_name, source=source_par, color="red")

    

    # in the plot 4 use followingimpo
    SIZES = list(range(6, 22, 3)) # for some sizes
    COLORS = Spectral5 # for some colors

    def plot_par():
        xs = par_df[x.value].values
        ys = par_df[y.value].values
        x_title = x.value.title()
        y_title = y.value.title()

        

        kw = dict()
           
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
            p4.xaxis.major_label_orientation = pd.np.pi / 4 # rotates labels... ugh. how about some datetime madness
                    
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


    print(columns)
    print(discrete)
    print(continuous)
    print(quantileable)

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
        fid=pd.DataFrame(polymer.getdata(ie),index=np.linspace(dw,dw*bs*nblk,bs*nblk),
                         columns=['real', 'im'])/ns
        fid['magnitude']=( fid['real']**2 + fid['im']**2 )**0.5 # last two lines may represent a seperate method
        fid=pd.DataFrame(polymer.getdata(ie),index=np.linspace(dw,dw*bs*nblk,bs*nblk),
                     columns=['real', 'im'])/ns
        fid['magnitude']=( fid['real']**2 + fid['im']**2 )**0.5
        source_fid.data = ColumnDataSource.from_df(fid)
        
        #
        #
        #
        #
        #
        
        try:
            tau = get_x_axis(parameters, nblk)
            startpoint=int(0.05*bs)-1
            endpoint=int(0.1*bs)
            phi=np.zeros(nblk)  
            for blk in range(nblk):
                start=startpoint + blk * bs
                end=endpoint + blk * bs
                phi[blk]=fid['magnitude'].iloc[start:end].sum() / (endpoint-startpoint)
            df = pd.DataFrame(data=np.c_[tau, phi], columns=['tau', 'phi'])

            df['phi_normalized'] = (df['phi'] - df['phi'].iloc[0] ) / (df['phi'].iloc[-1] - df['phi'].iloc[1] )

            fit_option = 2
            def model_func(t, A, K, C):
                return A * np.exp(- K * t) + C
            def fun(par,t,y):
                A, K, C = par
                return model_func(t, A,K,C) - y
            
        
            if fit_option ==1:
                from utils import fit_exp_linear
                C0 = 0 # offset
                popt = fit_exp_linear(df.tau, df.phi_normalized, C0)
            elif fit_option == 2:
                # needs prior knowledge for p0...
                from scipy.optimize import leastsq
                #popt, _ = curve_fit(model_func, np.array(df.tau), np.array(df.phi_normalized), p0=[1.0, 0.1**-1, 0.0])
                p0=[1.0, 0.1**-1, 0.0]
                popt, _ = leastsq(fun, p0  , args=(np.array(df.tau),np.array(df.phi_normalized)) )
            df['fit_phi'] = model_func(df.tau, *popt)
            df['phi_normalized'] = (df['phi'] - df['phi'].iloc[0] ) / (df['phi'].iloc[-1] - df['phi'].iloc[1] )

            fit_option = 2
            def model_func(t, A, K, C):
                return A * np.exp(- K * t) + C
            def fun(par,t,y):
                A, K, C = par
                return model_func(t, A,K,C) - y
            
        
            if fit_option ==1:
                from utils import fit_exp_linear
                C0 = 0 # offset
                popt = fit_exp_linear(df.tau, df.phi_normalized, C0)
            elif fit_option == 2:
                # needs prior knowledge for p0...
                from scipy.optimize import leastsq
                #popt, _ = curve_fit(model_func, np.array(df.tau), np.array(df.phi_normalized), p0=[1.0, 0.1**-1, 0.0])
                p0=[1.0, 0.1**-1, 0.0]
                popt, _ = leastsq(fun, p0  , args=(np.array(df.tau),np.array(df.phi_normalized)) )
            df['fit_phi'] = model_func(df.tau, *popt)
            source_df.data = ColumnDataSource.from_df(df)
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
    
    slider = Slider(start=1, end=nr_experiments+1, value=1, step=1,callback_policy='mouseup')
    slider.callback=CustomJS(args=dict(source=source),code="""
        source.data = { value: [cb_obj.value] }
    """)#unfortunately this customjs is needed to throttle the callback in current version of bokeh

    # choose the parameters for plot p3 in a dropdown menu
    #parameters given may vary in one file, merge the parameters
##    merge_par=parameters
##    for i in range(1,nr_experiments):
##        add_par=polymer.getparameter(i) #read the parameters
##        merge_par.update(add_par) #and merge them into one dictionary
##    
##    menu = [(key, "None") for key in merge_par.keys()] 
##    dropdown_x_par = Dropdown(label="parameter x-axis", menu=menu)    
##    dropdown_y_par = Dropdown(label="parameter y-axis", menu=menu)


    
    # try to add some select boxes for p4
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
    doc.add_root(column(slider, p1, p2,
                        #p3, dropdown_x_par, dropdown_y_par,
                        ))
    doc.add_root(layout_p4)
    doc.add_root(source) # i need to add the source for some reason...

bokeh_app = Application(FunctionHandler(modify_doc))

server = Server({'/': bokeh_app}, io_loop=io_loop)
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    io_loop.add_callback(server.show, "/")
io_loop.start()

