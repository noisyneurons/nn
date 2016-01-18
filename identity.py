# identity.py
# test/retest program results --> to benchmark other versions of the code...


from __future__ import division
import math, random
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd

from jorg.neuralnet_v1 import \
NeuralNet, Instance, NetworkDataCollector, weight_init_function_random, learning_rate_function

from jorg.activation_classes import SigmoidIO, LinearIO, ConstantOutput, GaussGauss, Gauss, STDNonMonotonicIOFunction

def learning_rate_function():
    return -1.0

def calc_n_hidden_layers(n_neurons_for_each_layer):
    n_hidden_layers = 0
    if len(n_neurons_for_each_layer) > 2:
        n_hidden_layers = len(n_neurons_for_each_layer) - 2
    return n_hidden_layers

def intermediate_post_process(trial_params, data_collector, dfs_concatenated):
    weight_series = data_collector.extract_weights(layer_number=1)

    #print "weight_series =", weight_series
    w0_series = weight_series[:,0,0]
    w0_series.name = "weight0"
    w1_series = weight_series[:,0,1]
    w1_series.name = "weight1"

    ratios = w0_series / w1_series
    convert_to_angles_function = lambda x: 180.0 * math.atan(x) / math.pi
    extracted_series = ratios.map(convert_to_angles_function)
    extracted_series.name = "hyperplane_angle"

    df = DataFrame([w0_series, w1_series, extracted_series])
    #, columns=["weight0", "weight1", "hyperplane_angle", "epochs"] )
    df["treatment"] = trial_params
    dfs_concatenated = pd.concat([dfs_concatenated, df])

    return dfs_concatenated

# "identity" training set
training_set = [ Instance( [0.0, 0.0], [0.0, 0.0] ), Instance( [0.0, 1.0], [0.0, 1.0] ), Instance( [1.0, 0.0], [1.0, 0.0] ), Instance( [1.0, 1.0], [1.0, 1.0] ) ]

n_inputs = 2
n_outputs = 2
n_neurons_for_each_layer = [n_inputs, 2, n_outputs]
n_hidden_layers = calc_n_hidden_layers(n_neurons_for_each_layer)

sigmoid = SigmoidIO()
linear = LinearIO()
constant = ConstantOutput()
nonmon = STDNonMonotonicIOFunction()

# specify neuron transforms, weight initialization, and learning rate functions... per layer

neurons_ios = [None] + [nonmon] * n_hidden_layers + [sigmoid]
weight_init_functions = [None] + [ weight_init_function_random ]*n_hidden_layers + [ weight_init_function_random ]
learning_rate_functions = [None] + [ learning_rate_function ]*n_hidden_layers + [ learning_rate_function ]

results = []
dfs_concatenated = DataFrame([])

for seed_value in range(3):
    print "seed = ", seed_value,
    random.seed(seed_value)
        
    # initialize the neural network
    network = NeuralNet(n_neurons_for_each_layer, neurons_ios, weight_init_functions, learning_rate_functions)
    
    data_collection_interval = 1000
    data_collector = NetworkDataCollector(network, data_collection_interval)
    
    # start training on test set one
    max_epochs = 30000
    epoch_and_MSE = network.backpropagation(training_set, 0.00001, max_epochs, data_collector)
    results.append(epoch_and_MSE[0])

    print "\n\nNet After Training\n", network

    # save the network
    network.save_to_file( "trained_configuration.pkl" )
    # load a stored network
    # network = NeuralNet.load_from_file( "trained_configuration.pkl" )
   
    dfs_concatenated = intermediate_post_process(seed_value, data_collector, dfs_concatenated)

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

print "keys=\t", dfs_concatenated.keys()


end_angle_values = dfs_concatenated["end"]["hyperplane_angle"]
treatment_values = dfs_concatenated["treatment"]["hyperplane_angle"]
list_of_dfs = [treatment_values] + [ (dfs_concatenated[epochs]["hyperplane_angle"]) for epochs in [0] ] + [end_angle_values]
selected_df =  pd.concat( list_of_dfs, axis=1 )

print selected_df
print type(selected_df)

plt.scatter(selected_df["treatment"], selected_df["end"])
pd.scatter_matrix(selected_df, diagonal='kde', color='k', alpha=0.3)
plt.show()
