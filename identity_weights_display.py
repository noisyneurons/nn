# identity_weights_display.py
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
NeuralNet, Instance, NetworkDataCollector, weight_init_function_random

from jorg.activation_classes import SigmoidIO, LinearIO, ConstantOutput, GaussGauss, Gauss, STDNonMonotonicIOFunction
sigmoid = SigmoidIO()
linear = LinearIO()
constant = ConstantOutput()
nonmon = STDNonMonotonicIOFunction()

n_hidden_neurons = 1
learning_rate = -0.5
rotation = 0.0
n_trials = 5
max_epochs = 3000
error_criterion = 0.00001
output_neurons_io_function = sigmoid
data_collection_interval = 1000
n_inputs = 2
n_outputs = 2
n_neurons_for_each_layer = [n_inputs, n_hidden_neurons, n_outputs]

def learning_rate_function():
    return learning_rate

def experiment_set_selected_weights(network):
    output_neurons = network.layers[2].neurons
    for output_neuron in output_neurons:
        output_neuron.links[0].weight = 0.1
        output_neuron.links[1].weight = 0.1

    hidden_neurons = network.layers[1].neurons_wo_bias_neuron
    for hidden_neuron in hidden_neurons:
        hidden_neuron.links[0].weight = 0.0
        hidden_neuron.links[1].weight = 1.0

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

def print_nets_io_map():
    for example_number, example in enumerate(training_set):
        inputs_for_training_example = example.features
        network.inputs_for_training_example = inputs_for_training_example
        output_from_network = network.calc_networks_output()
        print "\tnetworks input:", example.features, "\tnetworks output:", output_from_network, "\ttarget:", example.targets

def calc_n_hidden_layers(n_neurons_for_each_layer):
    n_hidden_layers = 0
    if len(n_neurons_for_each_layer) > 2:
        n_hidden_layers = len(n_neurons_for_each_layer) - 2
    return n_hidden_layers

n_hidden_layers = calc_n_hidden_layers(n_neurons_for_each_layer)

# "identity" training set
training_set = [ Instance( [0.0, 0.0], [0.0, 0.0] ), Instance( [0.0, 1.0], [0.0, 1.0] ), Instance( [1.0, 0.0], [1.0, 0.0] ), Instance( [1.0, 1.0], [1.0, 1.0] ) ]

# rotate
for an_instance in training_set:
    an_instance.rotate_by(rotation)

def post_process(trial_params, collection_function, data_frame):
    data_dictionary = collection_function(trial_params, layer_number=1)
    df_segment = DataFrame(data_dictionary)
    data_frame = pd.concat([data_frame, df_segment])
    return data_frame

# specify neuron transforms, weight initialization, and learning rate functions... per layer
neurons_ios = [None] + [nonmon] * n_hidden_layers + [output_neurons_io_function]
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
    ####

    # second_output_neuron = network.layers[-1].neurons[1]
    # second_output_neuron.activation_function = ConstantOutput()

    ####

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

    print network
    # print networks input - output function
    print_nets_io_map()

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









# plt.scatter(end_records["trial"], end_records["hyperplane_angle"])
# plt.show()
# plt.scatter(grouped_records.index, grouped_records["hyperplane_angle"])
# plt.show()


# plt.plot(grouped_records["hyperplane_angle"])
# plt.plot(result_records)

# treatment_values = df_weights["treatment"]["hyperplane_angle"]
# list_of_dfs = [treatment_values] + [ (df_weights[epochs]["hyperplane_angle"]) for epochs in [0] ] + [end_angle_values]
# selected_df =  pd.concat( list_of_dfs, axis=1 )
#
# print selected_df
# print type(selected_df)
#
# plt.scatter(selected_df["treatment"], selected_df["end"])
# #pd.scatter_matrix(selected_df, diagonal='kde', color='k', alpha=0.3)
# plt.show()
