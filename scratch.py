#temp
from __future__ import division


a = [1,2,3]
b = [2,4,6]

for i, value_pair in enumerate(zip(a,b)):
    print i
    aval, bval = value_pair
    print "aval=", aval, "bval=" ,  bval