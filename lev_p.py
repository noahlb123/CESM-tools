import os
import sys
import platform
from tools import ToolBox
from netCDF4 import Dataset
import numpy as np
T = ToolBox()

def evaluate(s):
    l = len(s)
    if s[l - 4:l] == ' && ':
        s = s[0:l - 4]
    os.system(s)
    return 'cd ' + root + ' && '

if len(sys.argv) < 4:
    raise Exception('3 command line arguments required: <root path> <file name> <target variable>')
root = sys.argv[1]
file_name = sys.argv[2]
target_v = sys.argv[3]

to_eval = 'cd ' + root + ' && '
to_eval += "ncap2 -O -s 'p=double(a*p0+b*ps);' " + file_name + ' ' + file_name + ' && '
evaluate(to_eval)