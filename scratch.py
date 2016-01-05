#temp
from __future__ import division


class Neuron:
    def __init__(self,x):
        self.x = x

    def print_it(self):
        print str(self.x)

a_list = [Neuron, 4]
g = a_list[0](33)

# g = Neuron(33)

g.print_it()
