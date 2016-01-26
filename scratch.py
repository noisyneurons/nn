#temp
from __future__ import division
import math, random
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd

from jorg.activation_classes import SigmoidIO, LinearIO, ConstantOutput, GaussGauss, Gauss, STDNonMonotonicIOFunction

sigmoid = SigmoidIO()
nonmon  = STDNonMonotonicIOFunction()

def findNetInputThatGeneratesMaximumOutput(io_function):
    grid_divisions = 20
    offset = 0.0
    increment = 5.0 / grid_divisions
    max_xy = None
    for i in xrange(40):
        xys = gen_testing_array(io_function, offset, increment, grid_divisions)
        max_xy = max(xys, key=lambda xy: xy[1])
        offset = max_xy[0] - increment
        increment = 2.0 * increment / grid_divisions
    return max_xy[0]


def gen_testing_array(io_function, offset, increment, grid_divisions):
    xs = [ (offset + (grid_div * increment)) for grid_div in range(0, grid_divisions)]
    return [ [x, io_function(x)] for x in xs ]

print findNetInputThatGeneratesMaximumOutput(nonmon.io)

input_vals = [0.01 * an_x for an_x in xrange(300)]
print input_vals

y = [ nonmon.io(input_val) for input_val in input_vals]



plt.scatter(input_vals, y)
plt.show()