# activation_classes
from __future__ import division
import math, random
from math import exp
from scipy.stats import norm
import numpy as np

class SigmoidIO:    
    def io(self, net_input):
        net_input_clipped = np.clip( net_input, -500, 500 )
        return 1.0 / (1.0 + exp(-net_input_clipped))
        
    def io_derivative(self, _, output):
        return (output * (1.0 - output))


class LinearIO:
    def io(self, net_input):
        return net_input
        
    def io_derivative(self, _, __):
        return 1.0


class ConstantOutput:
    def io(self, _):
        return 0.5

    def io_derivative(self, _, __):
        return 0.0


class FlatLine(ConstantOutput):
    def io(self, _):
        return 0.0

class Gauss:    
    def __init__(self, peak, std):    
        self.peak = peak
        self.std = std
        
    def bump_on_right(self, x):
        return norm.pdf( (x - self.peak) / self.std )

    def io(self, net_input):
        x = np.clip( net_input, -500, 500 )
        output =  self.bump_on_right(x)
        return output
        
    def io_derivative(self, _, output):
        return


class GaussGauss:    
    def __init__(self, peak, std):    
        self.peak = peak
        self.std = std
    
    def bump_on_left(self, x):
        return norm.pdf( (x + self.peak) / self.std )
        
    def bump_on_right(self, x):
        return norm.pdf( (x - self.peak) / self.std )

    def io(self, net_input):
        x = np.clip( net_input, -500, 500 )
        output =  self.bump_on_right(x) - self.bump_on_left(x)
        return output
        
    def io_derivative(self, _, output):
        return

class STDNonMonotonicIOFunction:
    # s is the horizontal shift of the io function, a is the amplitude of the "sidelobes," and w is the sidelobe "width"
    def __init__(self, s=4.0, a=0.5, w=1.0):
        self.s = s
        self.a = a
        self.w = w        

    def io(self, x):
        return 1.49786971589547 * (self.h(x) - 0.166192596930178)
    
    def h(self, x):
        return self.sig(x) + self.a * (1.0 - self.sig(x+self.s) - self.sig(x-self.s))
    
    def sig(self, x):
        return 1.0 / (1.0 + exp(-1.0 * x/self.w))
    
    def io_derivative(self, input, output):
        return 1.49786971589547 * self.j(input)
    
    def j(self, x):
        return (exp(-x) / ((exp(-x) + 1.0)**2.0)) - (self.a *  \
        (  self.exp_p_s(x)/(((self.exp_p_s(x) + 1.0)**2.0) * self.w)   +  \
        self.exp_m_s(x)/(((self.exp_m_s(x) + 1.0)**2.0) * self.w)    )) 
        
    def exp_p_s(self, x):
        return  exp(-(x+self.s)/self.w)

    def exp_m_s(self, x):
        return  exp(-(x-self.s)/self.w)


class NonmonSig:
    # s is the horizontal shift of the io function, a is the amplitude of the "sidelobes," and w is the sidelobe "width"
    def __init__(self, s=4.0, a=0.5, w=1.0):
        self.s = s
        self.a = a
        self.w = w

    def io(self, net_input):
        net_input_clipped = np.clip( net_input, -500, 500 )
        return 1.0 / (1.0 + exp(-net_input_clipped))

    def io_derivative(self, input, output):
        return 1.49786971589547 * self.j(input)

    def j(self, x):
        return (exp(-x) / ((exp(-x) + 1.0)**2.0)) - (self.a * (  self.exp_p_s(x)/(((self.exp_p_s(x) + 1.0)**2.0) * self.w) + self.exp_m_s(x)/(((self.exp_m_s(x) + 1.0)**2.0) * self.w)    ))

    def exp_p_s(self, x):
        return  exp(-(x+self.s)/self.w)

    def exp_m_s(self, x):
        return  exp(-(x-self.s)/self.w)

    
def findNetInputThatGeneratesMaximumOutput():
    grid_divisions = 20.0
    offset = 0.0
    increment = 5.0 / grid_divisions
    max_xy = None
    for i in xrange(40):   
        xys = gen_testing_array(offset, increment, grid_divisions)
        max_xy = max(xys, key=lambda xy: xy[1])
        offset = max_xy[0] - increment
        increment = 2.0 * increment / grid_divisions    
    return max_xy[0]

  
def gen_testing_array(offset, increment, grid_divisions):
    xs = [ (offset + (grid_div * increment)) for grid_div in range(0, grid_divisions)]
    return [ [x, io(x)] for x in xs ]
    

# def findNetInputThatGeneratesMaximumOutput(io_function):
#     grid_divisions = 20
#     offset = 0.0
#     increment = 5.0 / grid_divisions
#     max_xy = None
#     for i in xrange(40):
#         xys = gen_testing_array(io_function, offset, increment, grid_divisions)
#         max_xy = max(xys, key=lambda xy: xy[1])
#         offset = max_xy[0] - increment
#         increment = 2.0 * increment / grid_divisions
#     return max_xy[0]
#
#
# def gen_testing_array(io_function, offset, increment, grid_divisions):
#     xs = [ (offset + (grid_div * increment)) for grid_div in range(0, grid_divisions)]
#     return [ [x, io_function(x)] for x in xs ]
#
# print findNetInputThatGeneratesMaximumOutput(nonmon.io)

