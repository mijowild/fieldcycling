import numpy as np

from bokeh.layouts import row, widgetbox
from bokeh.models import CustomJS, Slider, Paragraph, Dialog, Button
from bokeh.plotting import figure, output_file, show, ColumnDataSource

x = np.linspace(0, 10, 500)
y = np.sin(x)

source = ColumnDataSource(data=dict(x=x, y=y))

plot = figure(y_range=(-10, 10), plot_width=400, plot_height=400)

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

descr_box = Paragraph(text='content loading...')

btn_close_loading = Button(label='Close Loading')
dialog_loading = Dialog(
    title='loading', content=row(descr_box), name='loading_dialog',
    buttons=[btn_close_loading], visible=True)



layout = vplot(
    dialog_loading,
    plot,
    widgetbox(btn_close_loading),
)

output_file("slider.html", title="slider.py example")

show(layout)
