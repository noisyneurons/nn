# main_sop_learning_stats.py  OR target function
# test/retest program to benchmark other versions of the code...
#
# results= [167, 172, 163, 170, 173, 170, 167, 174, 167, 169]
#
# 169.5

from __future__ import division
import math, random
import numpy as np   
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd

from jorg.elmannet import NeuralNet, Instance, NetworkDataCollector, \
intermediate_post_process, weight_init_function_random, learning_rate_function

from jorg.activation_classes import SigmoidIO, LinearIO, GaussGauss, Gauss, STDNonMonotonicIOFunction

def learning_rate_function():
    return -1.0
   
def experimental_weight_setting_function(network):
    hidden_layer = network.layers[0]

# "OR" training set
training_set = [ Instance( [0.0, 0.0], [0.0] ), Instance( [0.0, 1.0], [1.0] ), Instance( [1.0, 0.0], [1.0] ), Instance( [1.0, 1.0], [1.0] ) ]

n_inputs = 2
n_outputs = 1
n_hiddens = 0
n_hidden_layers = 0

# specify neuron transforms, weight initialization, and learning rate functions... per layer eg: [ hidden_layer_1, hidden_layer_2, output_layer ]
an_io_transform = SigmoidIO()
neurons_ios = [an_io_transform] + [ an_io_transform ]*n_hidden_layers + [ an_io_transform ] # [ SigmoidIO() ] #
weight_init_functions = [ weight_init_function_random ] + [ weight_init_function_random ]*n_hidden_layers + [ weight_init_function_random ]
learning_rate_functions = [ learning_rate_function ] + [ learning_rate_function ]*n_hidden_layers + [ learning_rate_function ]

results = []

for seed_value in range(10):
    print "seed = ", seed_value,
    random.seed(seed_value)
        
    # initialize the neural network
    network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, neurons_ios,
                        weight_init_functions, learning_rate_functions)
    
    experimental_weight_setting_function(network)
    
    data_collection_interval = 1000
    data_collector = NetworkDataCollector(network, data_collection_interval)
    
    # start training on test set one
    epoch_and_MSE = network.backpropagation(training_set, 0.01, data_collector)
    results.append(epoch_and_MSE[0])

    # save the network
    network.save_to_file( "trained_configuration.pkl" )
    # load a stored network
    network = NeuralNet.load_from_file( "trained_configuration.pkl" ) 
   
    # print out the result
    for instance in training_set:
        output_from_network = network.calc_networks_output(instance.features)
        input_to_neuron = network.layers[0].neurons[0].netinput
        print instance.features, "\tneurons input:", input_to_neuron, "\tnetworks output:", output_from_network, "\ttarget:", instance.targets
    
print results
print
print np.median(results)
print
