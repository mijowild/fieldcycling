from datetime import date
from random import randint
import glob
import os

from bokeh.io import output_file, show
from bokeh.layouts import widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn, Dropdown

output_file("data_table.html")


filemenu=glob.glob('testdata\\*.txt')
dropd=Dropdown(menu=[(file, None) for file in filemenu]
               + [('all *.txt in folder', None)])
data = dict(
        path=[os.path.join(os.getcwd(),'testdata') for i in range(10)],
        #file=[os.path.join(os.getcwd(),'testdata') for i in range(10)],
        #file=[dropd for i in range(10)],
        file=[filemenu for i in range(10)],
    )
source = ColumnDataSource(data)

columns = [
        TableColumn(field="path", title="Path"),
        TableColumn(field="file", title="File(s)"),
    ]
data_table = DataTable(source=source, columns=columns, width=600, height=380,
                       editable=True)


def cb(attr,old,new):
    print(new)
source.on_change('data',cb)

show(widgetbox(data_table,dropd))
