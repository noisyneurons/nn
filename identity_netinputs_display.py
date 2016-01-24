# identity_netinputs_display.py
# test/retest program results --> to benchmark other versions of the code...

#TODO implement plot (or just simple display) of netinputs for all examples
#TODO implement weight-sharing

from __future__ import division
import math, random
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd

from jorg.neuralnet_v1 import \
NeuralNet, Instance, NetworkDataCollector, weight_init_function_random, learning_rate_function

from jorg.activation_classes import SigmoidIO, LinearIO, ConstantOutput, GaussGauss, Gauss, STDNonMonotonicIOFunction

n_hidden_neurons = 1
learning_rate = -0.01
rotation = 0.0
n_trials = 2
max_epochs = 3000
error_criterion = 0.00001

def learning_rate_function():
    return learning_rate

def experiment_set_selected_weights(network):
    output_neurons = network.layers[2].neurons
    for output_neuron in output_neurons:
        output_neuron.links[0].weight = 0.0
        output_neuron.links[1].weight = 0.0

    hidden_neurons = network.layers[1].neurons_wo_bias_neuron
    for hidden_neuron in hidden_neurons:
        hidden_neuron.links[0].weight = 1.0
        hidden_neuron.links[1].weight = 0.0

# "identity" training set
training_set = [ Instance( [0.0, 0.0], [0.0, 0.0] ), Instance( [0.0, 1.0], [0.0, 1.0] ), Instance( [1.0, 0.0], [1.0, 0.0] ), Instance( [1.0, 1.0], [1.0, 1.0] ) ]

def calc_n_hidden_layers(n_neurons_for_each_layer):
    n_hidden_layers = 0
    if len(n_neurons_for_each_layer) > 2:
        n_hidden_layers = len(n_neurons_for_each_layer) - 2
    return n_hidden_layers


def intermediate_post_process_netinputs(trial_params, data_collector, dfs_concatenated):
    netinput_series = data_collector.extract_netinputs(layer_number=1)

    first_hidden_neuron_netinputs_series = netinput_series[:,0,:]
    first_hidden_neuron_netinputs_series.name = "first_hidden_neuron_netinputs_series"

    df = DataFrame([first_hidden_neuron_netinputs_series])
    #, columns=["weight0", "weight1", "hyperplane_angle", "epochs"] )
    #df["treatment"] = trial_params
    dfs_concatenated = pd.concat([dfs_concatenated, df])
    return dfs_concatenated


# rotate clockwise!!
for an_instance in training_set:
    an_instance.rotate_by(rotation)

n_inputs = 2
n_outputs = 2
n_neurons_for_each_layer = [n_inputs, n_hidden_neurons, n_outputs]
n_hidden_layers = calc_n_hidden_layers(n_neurons_for_each_layer)

sigmoid = SigmoidIO()
linear = LinearIO()
constant = ConstantOutput()
nonmon = STDNonMonotonicIOFunction()

# specify neuron transforms, weight initialization, and learning rate functions... per layer
neurons_ios = [None] + [nonmon] * n_hidden_layers + [linear]
weight_init_functions = [None] + [ weight_init_function_random ]*n_hidden_layers + [ weight_init_function_random ]
learning_rate_functions = [None] + [ learning_rate_function ]*n_hidden_layers + [ learning_rate_function ]

results = []
dfs_concatenated = DataFrame([])

for seed_value in range(n_trials):
    print "seed = ", seed_value,
    random.seed(seed_value)
        
    # initialize the neural network
    network = NeuralNet(n_neurons_for_each_layer, neurons_ios, weight_init_functions, learning_rate_functions)
    experiment_set_selected_weights(network)
    print "\n\nNet BEFORE Training\n", network
    
    data_collection_interval = 1000
    data_collector = NetworkDataCollector(network, data_collection_interval)
    
    # start training on test set one
    epoch_and_MSE = network.backpropagation(training_set, error_criterion, max_epochs, data_collector)  # sop call
    # epoch_and_MSE = network.backpropagation(training_set, 0.0000001, max_epochs, data_collector)
    results.append(epoch_and_MSE[0])

    #print "\n\nNet After Training\n", network

    # save the network
    network.save_to_file( "trained_configuration.pkl" )
    # load a stored network
    # network = NeuralNet.load_from_file( "trained_configuration.pkl" )
   
    dfs_concatenated = intermediate_post_process_netinputs(seed_value, data_collector, dfs_concatenated)

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
print dfs_concatenated
print

print dfs_concatenated["end"][0]

# end_netinputs = dfs_concatenated["end"]["first_hidden_neuron_netinputs_series"]
# treatment_values = dfs_concatenated["treatment"]["first_hidden_neuron_netinputs_series"]
# list_of_dfs = [treatment_values] + [ (dfs_concatenated[epochs]["first_hidden_neuron_netinputs_series"]) for epochs in [0] ] + [end_netinputs]
# selected_df =  pd.concat( list_of_dfs, axis=1 )
#
# print selected_df
# print type(selected_df)

# plt.scatter(selected_df["treatment"], selected_df["end"])
# #pd.scatter_matrix(selected_df, diagonal='kde', color='k', alpha=0.3)
# plt.show()
