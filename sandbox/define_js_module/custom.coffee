# These are similar to python imports. BokehJS vendors its own versions
# of Underscore and JQuery. They are available as show here.
import * as _ from "underscore"
import * as $ from "jquery"

# The "core/properties" module has all the property types
import * as p from "core/properties"

# We will subclass in JavaScript from the same class that was subclassed
# from in Python
import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"

# This model will actually need to render things, so we must provide
# view. The LayoutDOM model has a view already, so we will start with that
export class CustomView extends LayoutDOMView

  initialize: (options) ->
    super(options)

    @render()

    # Set Backbone listener so that when the Bokeh slider has a change
    # event, we can process the new data
    @listenTo(@model.slider, 'change', () => @render())

  render: () ->
    # Backbone Views create <div> elements by default, accessible as @$el.
    # Many Bokeh views ignore this default <div>, and instead do things
    # like draw to the HTML canvas. In this case though, we change the
    # contents of the <div>, based on the current slider value.
    @$el.html("<h1>#{ @model.text }: #{ @model.slider.value }</h1>")
    @$el.find('h1').css({ 'color': '#686d8e', 'background-color': '#2a3153' })

export class Custom extends LayoutDOM

  # If there is an associated view, this is boilerplate.
  default_view: CustomView

  # The ``type`` class attribute should generally match exactly the name
  # of the corresponding Python class.
  type: "Custom"

  # The @define block adds corresponding "properties" to the JS model. These
  # should basically line up 1-1 with the Python model class. Most property
  # types have counterparts, e.g. bokeh.core.properties.String will be
  # p.String in the JS implementation. Where the JS type system is not yet
  # as rich, you can use p.Any as a "wildcard" property type.
  @define {
    text:   [ p.String ]
    slider: [ p.Any    ]
  }