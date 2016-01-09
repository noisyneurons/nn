# main_3_examples_right_angle_hyperplane.py
# program that demonstrates ability of non-mon io function to learn "best" hyperplane
#
# test/retest program results --> to benchmark other versions of the code...
#  test results have been moved to the bottom of this file, because they are so long.

from __future__ import division
import math, random
import numpy as np   
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd

from jorg.elmannet import NeuralNet, Instance, NetworkDataCollector, \
intermediate_post_process, weight_init_function_random, learning_rate_function

from jorg.activation_classes import STDNonMonotonicIOFunction

def learning_rate_function():
    return -10.0       

def experimental_weight_setting_function(network):
    pass

# training set
training_set = [ Instance( [0.0, 0.0], [0.0] ), Instance( [0.0, 1.0], [1.0] ), Instance( [1.0, 1.0], [1.0] ) ]

n_inputs = 2
n_outputs = 1
n_hiddens = 0
n_neurons_for_each_layer = [n_inputs, n_outputs]

n_hidden_layers = 0
if len(n_neurons_for_each_layer) > 2:
    n_hidden_layers = len(n_neurons_for_each_layer) - 2

# specify neuron transforms, weight initialization, and learning rate functions... per layer
an_io_transform = STDNonMonotonicIOFunction()
neurons_ios = [an_io_transform] + [ an_io_transform ]*n_hidden_layers + [ an_io_transform ] # [ SigmoidIO() ] #
weight_init_functions = [ weight_init_function_random ] + [ weight_init_function_random ]*n_hidden_layers + [ weight_init_function_random ]
learning_rate_functions = [ learning_rate_function ] + [ learning_rate_function ]*n_hidden_layers + [ learning_rate_function ]

results = []
dfs_concatenated = DataFrame([])

for seed_value in range(10):
    print "seed = ", seed_value,
    random.seed(seed_value)
        
    # initialize the neural network
    network = NeuralNet(n_neurons_for_each_layer, neurons_ios, weight_init_functions, learning_rate_functions, n_delays=3)

    print "\n\nNetwork State just after creation\n", network

    data_collector = NetworkDataCollector(network, data_collection_interval=1000)
    
    # start training on test set one
    epoch_and_MSE = network.backpropagation(training_set, error_limit=0.0000001, max_epochs=6000, data_collector=data_collector)

    results.append(epoch_and_MSE[0])
        
    dfs_concatenated = intermediate_post_process(seed_value, data_collector, dfs_concatenated)

    # print out the result
    for instance in training_set:
        output_from_network = network.calc_networks_output()
        input_to_neuron = network.layers[1].neurons[0].netinput
        print instance.features, "\tneurons input:", input_to_neuron, "\tnetworks output:", output_from_network, "\ttarget:", instance.targets
    
print results
print
print np.median(results)
print
print dfs_concatenated
print

end_angle_values = dfs_concatenated["end"]["hyperplane_angle"]
treatment_values = dfs_concatenated["treatment"]["hyperplane_angle"]
list_of_dfs = [treatment_values] + [ (dfs_concatenated[epochs]["hyperplane_angle"]) for epochs in [0] ] + [end_angle_values]
selected_df =  pd.concat( list_of_dfs, axis=1 )

print selected_df
print type(selected_df)

plt.scatter(selected_df["treatment"], selected_df["end"])
pd.scatter_matrix(selected_df, diagonal='kde', color='k', alpha=0.3)
plt.show()

# test/retest program results --> to benchmark other versions of the code...
# [2784, 2334, 2870, 2897, 2945, 2685, 2782, 2947, 2421, 2929]
# 2827.0
#
# epochs                    0      1000       end      2000  treatment
# weight0            0.344422  0.189813  0.122256  0.141979          0
# weight1            0.257954  4.694845  4.771335  4.749801          0
# hyperplane_angle  53.168661  2.315213  1.467768  1.712155          0
# weight0           -0.365636  0.141482  0.117166  0.122763          1
# weight1            0.347434  4.710894  4.769056  4.760050          1
# hyperplane_angle -46.462231  1.720245  1.407356  1.477345          1
# weight0            0.456034  0.199489  0.122485  0.145337          2
# weight1            0.447827  4.688437  4.771447  4.747015          2
# hyperplane_angle  45.520214  2.436411  1.470487  1.753646          2
# weight0           -0.262035  0.202408  0.122517  0.146352          3
# weight1            0.044229  4.686092  4.771470  4.746074          3
# hyperplane_angle -80.419290  2.473257  1.470864  1.766240          3
# weight0           -0.263952  0.207435  0.122548  0.148135          4
# weight1           -0.396834  4.681352  4.771477  4.744284          4
# hyperplane_angle  33.629644  2.537172  1.471232  1.788417          4
# weight0            0.122902  0.178567  0.121766  0.137983          5
# weight1            0.241787  4.700429  4.771073  4.752588          5
# hyperplane_angle  26.944488  2.175594  1.461974  1.663018          5
# weight0            0.293340  0.189670  0.122266  0.141930          6
# weight1            0.321954  4.694925  4.771316  4.749838          6
# hyperplane_angle  42.337406  2.313436  1.467901  1.711543          6
# weight0           -0.176167  0.207634  0.122548  0.148207          7
# weight1           -0.349151  4.681145  4.771478  4.744209          7
# hyperplane_angle  26.773645  2.539710  1.471224  1.789308          7
# weight0           -0.273294  0.150127  0.118837  0.126713          8
# weight1            0.462295  4.709189  4.769705  4.758447          8
# hyperplane_angle -30.590231  1.825941  1.427225  1.525371          8
# weight0           -0.036993  0.205757  0.122536  0.147534          9
# weight1           -0.126688  4.683041  4.771484  4.744907          9
# hyperplane_angle  16.277690  2.515761  1.471082  1.780932          9
#
#                   treatment          0       end
# hyperplane_angle          0  53.168661  1.467768
# hyperplane_angle          1 -46.462231  1.407356
# hyperplane_angle          2  45.520214  1.470487
# hyperplane_angle          3 -80.419290  1.470864
# hyperplane_angle          4  33.629644  1.471232
# hyperplane_angle          5  26.944488  1.461974
# hyperplane_angle          6  42.337406  1.467901
# hyperplane_angle          7  26.773645  1.471224
# hyperplane_angle          8 -30.590231  1.427225
# hyperplane_angle          9  16.277690  1.471082
# <class 'pandas.core.frame.DataFrame'>
