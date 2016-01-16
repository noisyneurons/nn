# main_test_neuralnet_v1_w_hidden_layer.py
# main_sop_hidden_layer_learning_stats.py   XOR target function
# test/retest program results --> to benchmark other versions of the code...
#
# results= [1235, 971, 935, 927, 824, 820, 780, 1088, 732, 1034]
#
# 931.0

from __future__ import division
import math, random
import numpy as np

from jorg.neuralnet_v1 import \
NeuralNet, Instance, NetworkDataCollector, weight_init_function_random, learning_rate_function

from jorg.activation_classes import SigmoidIO, LinearIO, GaussGauss, Gauss, STDNonMonotonicIOFunction

def learning_rate_function():
    return -1.0

# "XOR" training set
training_set = [ Instance( [0.0, 0.0], [0.0] ), Instance( [0.0, 1.0], [1.0] ), Instance( [1.0, 0.0], [1.0] ), Instance( [1.0, 1.0], [0.0] ) ]

n_inputs = 2
n_outputs = 1
n_neurons_for_each_layer = [n_inputs, 3, n_outputs]

n_hidden_layers = 0
if len(n_neurons_for_each_layer) > 2:
    n_hidden_layers = len(n_neurons_for_each_layer) - 2

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
    network = NeuralNet(n_neurons_for_each_layer, neurons_ios, weight_init_functions, learning_rate_functions)

    #print "\n\nNetwork State just after creation\n", network
    
    data_collection_interval = 1000
    data_collector = NetworkDataCollector(network, data_collection_interval)
    
    # start training on test set one
    epoch_and_MSE = network.backpropagation(training_set, 0.01, 3000, data_collector)
    results.append(epoch_and_MSE[0])

    # save the network
    network.save_to_file( "trained_configuration.pkl" )
    # load a stored network
    # network = NeuralNet.load_from_file( "trained_configuration.pkl" )
   
    print "\n\nNetwork State after backpropagation\n", network, "\n"

    # print out the result
    for example_number, example in enumerate(training_set):
        inputs_for_training_example = example.features
        network.inputs_for_training_example = inputs_for_training_example
        output_from_network = network.calc_networks_output()
        print "\tnetworks input:", example.features, "\tnetworks output:", output_from_network, "\ttarget:", example.targets

print results
print
print np.median(results)
print
