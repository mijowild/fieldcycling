import os
from bokeh.io import vplot
import pandas as pd
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.models.widgets import Button
from bokeh.plotting import figure, output_file, show

output_file("load_data_buttons.html")
file1=os.path.join(os.getcwd(),'testdata','data_file_1.txt')
df1 = pd.read_csv(file1)
file2=os.path.join(os.getcwd(),'testdata','data_file_2.txt')
df2 = pd.read_csv(file2)

plot = figure(plot_width=400, plot_height=400)


source = ColumnDataSource(data=ColumnDataSource.from_df(df1))
source2 = ColumnDataSource(data=ColumnDataSource.from_df(df2))

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

#the following lines are supposed to replace the JS code below in order to load the data only after toggling the buttons. somehow it doesnt work for me
##        $(document).ready(function() {
##            $.ajax({
##                type: "GET",
##                url: "data.txt",
##                dataType: "text",
##                success: function(data) {processData(data);}
##             });
##        });
##
##        function processData(allText) {
##            var allTextLines = allText.split(/\r\n|\n/);
##            var headers = allTextLines[0].split(',');
##            var lines = [];
##
##            for (var i=1; i<allTextLines.length; i++) {
##                var data = allTextLines[i].split(',');
##                if (data.length == headers.length) {
##
##                    var tarr = [];
##                    for (var j=0; j<headers.length; j++) {
##                        tarr.push(headers[j]+":"+data[j]);
##                    }
##                    lines.push(tarr);
##                }
##            }
##            // alert(lines);
##        }

callback = CustomJS(args=dict(source=source, source2=source2), code="""

        var data = source.get('data');
        var data2 = source2.get('data');
        data['x'] = data2['x' + cb_obj.get("name")];
        data['y'] = data2['y' + cb_obj.get("name")];
        source.trigger('change');

    """)

toggle1 = Button(label="Load data file 1", callback=callback, name="1")
toggle2 = Button(label="Load data file 2", callback=callback, name="2")

layout = vplot(toggle1, toggle2, plot)

show(layout)
