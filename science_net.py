# science_net.py
# test/retest program results --> to benchmark other versions of the code...
#
# results using linear outputs =  [53, 67, 59, 58, 76, 77, 57, 59, 59, 88]  median = 59.0
# results using sigmoid outputs = [167, 171, 153, 163, 149, 183, 162, 152, 181, 167]   median = 165.0
#

from __future__ import division
import math, random
import numpy as np

from jorg.neuralnet_v1 import \
NeuralNet, Instance, NetworkDataCollector, weight_init_function_random, learning_rate_function

from jorg.activation_classes import SigmoidIO, LinearIO, ConstantOutput, GaussGauss, Gauss, STDNonMonotonicIOFunction

def learning_rate_function():
    return -1.0

def replace_ios(network, neurons_io):
    output_neurons = network.layers[-1].neurons
    output_neurons[2].activation_function = neurons_io

# "identity" training set
training_set = [ Instance( [0.0, 0.0], [0.0, 0.0, 0.0] ), Instance( [0.0, 1.0], [0.0, 1.0, 0.0] ), Instance( [1.0, 0.0], [1.0, 0.0, 0.0] ), Instance( [1.0, 1.0], [1.0, 1.0, 1.0] ) ]

n_inputs = 2
n_outputs = 3
n_neurons_for_each_layer = [n_inputs, 6, n_outputs]

n_hidden_layers = 0
if len(n_neurons_for_each_layer) > 2:
    n_hidden_layers = len(n_neurons_for_each_layer) - 2

# specify neuron transforms, weight initialization, and learning rate functions... per layer eg: [ hidden_layer_1, hidden_layer_2, output_layer ]
sigmoid = SigmoidIO()
linear = LinearIO()
constant = ConstantOutput()

neurons_ios = [None] + [sigmoid] * n_hidden_layers + [sigmoid]
weight_init_functions = [None] + [ weight_init_function_random ]*n_hidden_layers + [ weight_init_function_random ]
learning_rate_functions = [None] + [ learning_rate_function ]*n_hidden_layers + [ learning_rate_function ]

results = []

for seed_value in range(10):
    print "seed = ", seed_value,
    random.seed(seed_value)
        
    # initialize the neural network
    network = NeuralNet(n_neurons_for_each_layer, neurons_ios, weight_init_functions, learning_rate_functions)

    replace_ios(network, ConstantOutput())
    #print "\n\nNetwork State just after creation\n", network
    
    data_collection_interval = 1000
    data_collector = NetworkDataCollector(network, data_collection_interval)
    
    # start training on test set one
    epoch_and_MSE = network.backpropagation(training_set, 0.01, 300, data_collector)
    replace_ios(network, SigmoidIO())
    epoch_and_MSE = network.backpropagation(training_set, 0.001, 600, data_collector)
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
