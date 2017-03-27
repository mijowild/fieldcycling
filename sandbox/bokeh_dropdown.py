from bokeh.io import output_file, show
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Dropdown
from bokeh.plotting import curdoc

menu = [("Quaterly", "time_windows"), ("Half Yearly", "time_windows"), None, ("Yearly", "time_windows")]
dropdown = Dropdown(label="Time Period", button_type="warning", menu=menu)

def function_to_call(attr, old, new):
    print (dropdown.value)

dropdown.on_change('value', function_to_call)

curdoc().add_root(dropdown)
