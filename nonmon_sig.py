# nonmon_sig.py

#TODO implement weight-sharing

from __future__ import division
import math, random
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd

from jorg.neuralnet_v1 import \
NeuralNet, Instance, NetworkDataCollector, weight_init_function_random

from jorg.activation_classes import SigmoidIO, LinearIO, ConstantOutput, GaussGauss, Gauss, STDNonMonotonicIOFunction, NonmonSig
sigmoid = SigmoidIO()
linear = LinearIO()
constant = ConstantOutput()
nonmon = STDNonMonotonicIOFunction()
nonmon_sig = NonmonSig()

n_hidden_neurons = 2
learning_rate = -1.0
weight_init_max = 0.5
rotation = 0.0
n_trials = 10
max_epochs = 3000
error_criterion = 0.001
hidden_neurons_io_function = nonmon
output_neurons_io_function = nonmon
data_collection_interval = 1000
n_inputs = 2
n_outputs = 1
n_neurons_for_each_layer = [n_inputs, n_hidden_neurons, n_outputs]

def calc_n_hidden_layers(n_neurons_for_each_layer):
    n_hidden_layers = 0
    if len(n_neurons_for_each_layer) > 2:
        n_hidden_layers = len(n_neurons_for_each_layer) - 2
    return n_hidden_layers
n_hidden_layers = calc_n_hidden_layers(n_neurons_for_each_layer)

def learning_rate_function():
    return learning_rate

def weight_init_function_random():
    return random.uniform(-weight_init_max, weight_init_max)

def experiment_set_selected_weights(network):
    pass

def create_4_graphs():
    fig = plt.figure()
    panel1 = fig.add_subplot(2, 2, 1)
    panel2 = fig.add_subplot(2, 2, 2)
    panel3 = fig.add_subplot(2, 2, 3)
    panel4 = fig.add_subplot(2, 2, 4)
    return panel1, panel2, panel3, panel4

def label_plot(panel,xl,yl):
    panel.set_xlabel(xl)
    panel.set_ylabel(yl)

def print_networks_io_map():
    for example_number, example in enumerate(training_set):
        inputs_for_training_example = example.features
        network.inputs_for_training_example = inputs_for_training_example
        output_from_network = network.calc_networks_output()
        print "\tnetworks input:", example.features, "\tnetworks output:", output_from_network, "\ttarget:", example.targets

# "identity" training set
# training_set = [ Instance( [0.0, 0.0], [0.0, 0.0] ), Instance( [0.0, 1.0], [0.0, 1.0] ), Instance( [1.0, 0.0], [1.0, 0.0] ), Instance( [1.0, 1.0], [1.0, 1.0] ) ]

# "AND" training set
# training_set = [ Instance( [0.0, 0.0], [0.0] ), Instance( [0.0, 1.0], [0.0] ), Instance( [1.0, 0.0], [0.0] ), Instance( [1.0, 1.0], [1.0] ) ]

# "normed AND" training set
training_set = [ Instance( [-0.5, -0.5], [0.0] ), Instance( [-0.5, 0.5], [0.0] ), Instance( [0.5, -0.5], [0.0] ), Instance( [0.5, 0.5], [1.0] ) ]

# "XOR" training set
# training_set = [ Instance( [0.0, 0.0], [0.0] ), Instance( [0.0, 1.0], [1.0] ), Instance( [1.0, 0.0], [1.0] ), Instance( [1.0, 1.0], [0.0] ) ]

# rotate
for an_instance in training_set:
    an_instance.rotate_by(rotation)

def post_process(trial_params, collection_function, data_frame):
    data_dictionary = collection_function(trial_params, layer_number=1)
    df_segment = DataFrame(data_dictionary)
    data_frame = pd.concat([data_frame, df_segment])
    return data_frame

# specify neuron transforms, weight initialization, and learning rate functions... per layer
neurons_ios = [None] + [hidden_neurons_io_function] * n_hidden_layers + [output_neurons_io_function]
weight_init_functions = [None] + [ weight_init_function_random ]*n_hidden_layers + [ weight_init_function_random ]
learning_rate_functions = [None] + [ learning_rate_function ]*n_hidden_layers + [ learning_rate_function ]

results = []
df_weights = DataFrame([])
df_netinputs = DataFrame([])

for seed_value in range(n_trials):
    print "seed = ", seed_value,
    random.seed(seed_value)
        
    # initialize the neural network
    network = NeuralNet(n_neurons_for_each_layer, neurons_ios, weight_init_functions, learning_rate_functions)
    experiment_set_selected_weights(network)

    print "\n\nNet BEFORE Training\n", network

    data_collector = NetworkDataCollector(network, data_collection_interval)
    
    # start training on test set one
    epoch_and_MSE = network.backpropagation(training_set, error_criterion, max_epochs, data_collector)  # sop call
    # epoch_and_MSE = network.backpropagation(training_set, 0.0000001, max_epochs, data_collector)
    results.append(epoch_and_MSE[0])

    # save the network
    network.save_to_file( "trained_configuration.pkl" )
    # load a stored network
    # network = NeuralNet.load_from_file( "trained_configuration.pkl" )

    df_weights = post_process(seed_value, data_collector.extract_weights, df_weights)
    df_netinputs = post_process(seed_value, data_collector.extract_netinputs, df_netinputs)

    print "\n\nNet AFTER Training\n", network, "\n"

    # print networks input-output function
    print_networks_io_map()

print results
print
print np.median(results)
print
print "df_weights:\n", df_weights
print "\ndf_netinputs:\n", df_netinputs
print

dfs = df_weights
end_records = dfs[dfs["epochs"] == "end"]
grouped_records = end_records.groupby("trial").mean()
result_records = grouped_records['hyperplane_angle']

panel1, panel2, panel3, panel4 = create_4_graphs()
label_plot(panel1, 'Trial Number', 'Hyperplane Angle (degrees)')
panel1.plot(result_records, 'go') # linestyle='--', color='g')

dfs = df_netinputs
end_records = dfs[dfs["epochs"] == "end"]
label_plot(panel2, 'Example Number', 'Netinput to Neuron')
panel2.scatter(end_records["example_number"], end_records["netinput"])
label_plot(panel3, 'Example Number', 'Output of Neuron')
panel3.scatter(end_records["example_number"], end_records["output"])

print "result_records:\n", result_records
print  "end_records", end_records
plt.show()
