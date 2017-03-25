import os
import sys
import numpy
import re
import stelardatafile as sdf

polymer=sdf.StelarDataFile('297K.sdf',r'../data')
polymer.sdfimport()
print(polymer.getparameter(140))
pe=sdf.StelarDataFile('100kHz-melting.sdf',r'../data')
pe.sdfimport()
print(polymer.getparameter(1000))
