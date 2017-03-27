from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import Slider

from custom import Custom

slider = Slider(start=0, end=10, step=0.1, value=0, title="value")

custom = Custom(text="Special Slider Display", slider=slider)

layout = column(slider, custom)

show(layout)
