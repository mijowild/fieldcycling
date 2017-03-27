from bokeh.core.properties import String, Instance
from bokeh.models import LayoutDOM, Slider

class Custom(LayoutDOM):

    __implementation__ = "custom.coffee"

    text = String(default="Custom text")

    slider = Instance(Slider)
