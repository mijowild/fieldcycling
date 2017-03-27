from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.models.callbacks import CustomJS
from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets import Slider

# this is the real callback that we want to happen on slider mouseup
def cb(attr, old, new):
    print("UPDATE", source.data['value'])
    #p.x_range=range(0, int(source.data['value']))

# This data source is just used to communicate / trigger the real callback
source = ColumnDataSource(data=dict(value=[]))
source.on_change('data', cb)

# a figure, just for example
p = figure(x_range=(0,1), y_range=(0,1))

# add a slider with a CustomJS callback and a mouseup policy to update the source
slider = Slider(start=1, end=10, value=1, step=0.1, callback_policy='mouseup')
slider.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")

curdoc().add_root(column(slider, p))

# make sure to add the source explicitly
curdoc().add_root(source)
