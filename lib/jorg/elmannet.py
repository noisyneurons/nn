# elmannet.py

# %matplotlib inline
from __future__ import division

import math
import random
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

def weight_init_function_random():
    return random.uniform(-0.5,0.5)

def learning_rate_function():
    return -1.0

def connect_to_next_layer(lower_layer, upper_layer):
    for upper_neuron in upper_layer.learning_neurons:
        for lower_neuron in lower_layer.neurons:
            a_link = Link(upper_neuron.weight_init_function, upper_neuron.network)
            upper_neuron.links.append(a_link)
            a_link.set_source_neuron(lower_neuron)
            lower_neuron.output_links.append(a_link)

def connect_within_layer(layer):
    for neuron in layer.learning_neurons:
        other_learning_neurons_in_layer = layer.learning_neurons - neuron
        for another_neuron in other_learning_neurons_in_layer:
            a_link = Link(neuron.weight_init_function, neuron.network)
            neuron.links.append(a_link)
            a_link.set_source_neuron(another_neuron)
            another_neuron.output_links.append(a_link)


def error_func(output, target):
    #if target == 0.555:
        #return 0.0
    return output - target


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

class NeuralNet:
    def __init__(self, n_neurons_for_each_layer, neurons_ios, weight_init_functions, learning_rate_functions, n_delays):

        self.n_neurons_for_each_layer = n_neurons_for_each_layer
        self.n_inputs = n_neurons_for_each_layer[0]
        self.n_outputs = n_neurons_for_each_layer[-1]
        self.n_hidden_layers = 0
        self.hidden_layers_present = False
        if len(n_neurons_for_each_layer) > 2:
            self.n_hidden_layers = len(n_neurons_for_each_layer) - 2
            self.hidden_layers_present = True

        self.layers = []
        self.upper_layers = None
        self.lower_layers = None
        self.upper_layers_in_reverse_order =  None
        self.lower_layers_in_reverse_order =  None

        self.neurons_ios = neurons_ios
        self.weight_init_functions = weight_init_functions
        self.learning_rate_functions = learning_rate_functions

        self.n_training_examples = None

        self.epoch = None
        self.example_number = None
        self.inputs_for_training_example = None
        self.not_done = True
        self.nearly_done = False

        # Elman delay...
        self.n_delays = n_delays

        # Do not touch
        self._create_network()
        self._n_links = None

    def _create_network(self):
        # modified to create a wider range of networks, inclucing Elman-type networks...

        # create input layer; i represents the 'current' layer number
        i = 0
        self.layers.append( InputNeuronLayer( i, self) )

        if self.hidden_layers_present:
        # create hidden layers
            for ii in range(1, (self.n_hidden_layers + 1)):
                self.layers.append(HiddenNeuronLayer(ii, self))

        # create output layer
        i = self.n_hidden_layers + 1
        self.layers.append( OutputNeuronLayer( i, self) )

        self.upper_layers = [aLayer for aLayer in self.layers[1:] ]
        self.lower_layers = [aLayer for aLayer in self.layers[:-1]]
        # Make subsequent backward error propagation code easier to write/understand!
        self.upper_layers_in_reverse_order =  [aLayer for aLayer in reversed(self.layers[1:])]
        self.lower_layers_in_reverse_order =  [aLayer for aLayer in reversed(self.layers[:-1])]

        # connect layers and... (depending on specific architecture, e.g. Elman arch)
        for lower_layer, upper_layer in zip(self.lower_layers, self.upper_layers):
            connect_to_next_layer(lower_layer, upper_layer)

        #TODO  add code here or above previous block: ... to add links from 1st hidden layer to Elman neurons in input layer (the '0th layer').



    def get_links(self):
        links = []
        for layer in self.layers:
            for neuron in layer.learning_neurons:
                links += neuron.links
        return links

    @property
    def n_links(self):
        if not self._n_links:
            self._n_links = len(self.get_links())
        return self._n_links

    def set_links(self, links):
        stop = 0
        for layer in self.layers:
            for neuron in layer.learning_neurons:
                start, stop = stop, (stop + len(neuron.links))
                neuron.links = links[start:stop]
        return self

    def backpropagation(self, training_set, error_limit, max_epochs, data_collector):
        self.n_training_examples = len(training_set)
        training_inputs  =  [instance.features for instance in training_set]
        training_targets =  [instance.targets for instance in training_set]

        MSE      = 1000.0 # any large enough number is good enough!
        self.epoch = 0
        self.not_done = True
        self.nearly_done = False

        # epoch loop
        while self.not_done:

            if self.epoch > max_epochs:
                self.nearly_done = True
                return self.epoch, MSE

            collected_output_errors = []

            # 1-training-example per iteration
            for example_number, inputs_for_training_example in enumerate(training_inputs):

                self.example_number = example_number
                self.inputs_for_training_example = inputs_for_training_example
                network_outputs = self.calc_networks_output()
                collected_output_errors += self.calc_output_neurons_errors(network_outputs, training_targets[example_number])
                self.calc_other_neurons_errors()
                self.adapt_all_layers()

                data_collector.store(self.epoch, example_number)

            sse = np.dot(collected_output_errors, collected_output_errors)
            MSE = sse / (self.n_outputs * self.n_training_examples)
            if self.nearly_done:
                self.not_done = False
            if MSE < error_limit:
                self.nearly_done = True
            self.epoch += 1

        return self.epoch, MSE


    def calc_networks_output(self):
        for layer in self.layers:
            for neuron in layer.neurons_wo_bias_neuron:
                neuron.calc_neurons_output()
        return [neuron.output for neuron in self.layers[-1].neurons ]


    def calc_output_neurons_errors(self, network_outputs, training_targets):
        # determine error at network's output
        # errors = [ error_func(aNetOutput, aTrainingTarget) for aNetOutput, aTrainingTarget in zip(network_outputs, training_targets) ]
        errors = [ (aNetOutput - aTrainingTarget) for aNetOutput, aTrainingTarget in zip(network_outputs, training_targets) ]
        # then calc & store errors in output neurons
        output_neurons = self.layers[-1].neurons
        for output_neuron, error in zip(output_neurons, errors):
            output_neuron.error = error
            output_neuron.calc_neurons_input_error()
        return errors

    def calc_other_neurons_errors(self):
        for lower_layer, upper_layer in zip(self.lower_layers_in_reverse_order, self.upper_layers_in_reverse_order):
            for neuron in lower_layer.learning_neurons:   #.learning_neurons.. We don't need to calculate bias neuron's error!!
                neuron.calc_neurons_error(upper_layer)

    def change_first_upper_layer_links(self, inputs_to_layer):
        # The first hidden layer receives its inputs externally, not from lower layer neurons
        for neuron in self.layers[1].learning_neurons:
            neuron.adapt_weights_in_links()

    def adapt_all_layers(self):
        # self.change_input_layer_links()
        for layer in self.layers:
            for neuron in layer.learning_neurons:
                neuron.adapt_weights_in_links()


    #TODO needs to be modified for true Elman
    # def change_input_layer_links(self):
    #     lower_layer_neuron_outputs = [neuron.output for neuron in self.layers[0].neurons]
    #     for neuron in self.layers[0].learning_neurons:
    #         neuron.adapt_weights_in_links()

    # #TODO This assumes a 1-to-1 Elman recurrent connections
    # def change_input_layer_links(self):
    #     for ff_neuron, elman_neuron in zip(self.layers[0].ff_neurons:
    #
    #         neuron.change_links(ff_neuron.output)
    #
    #     lower_layer_neuron_outputs = [neuron.output for neuron in self.layers[0].ff_neurons]
    #     for neuron in self.layers[0].learning_neurons:
    #             neuron.change_links(lower_layer_neuron_outputs)
    #

    def save_network_in_dictionary(self):
        links = self.get_links()  # need this to be called before "self.no_links" -- need to CLEAN this smell!
        return {
            "n_inputs"              : self.n_inputs,
            "n_outputs"             : self.n_outputs,
            "n_neurons_for_each_layer"             : self.n_neurons_for_each_layer,
            "n_hidden_layers"         : self.n_hidden_layers,
            "hidden_layers_present"  : self.hidden_layers_present,
            "neurons_ios"            : self.neurons_ios,
            "weight_init_functions"     : self.weight_init_functions,
            "learning_rate_functions"   : self.learning_rate_functions,
            "n_links"               : self.n_links,
            "links"                 : links
        }

    def save_to_file(self, filename = "network.pkl" ):
        import cPickle
        network_saved_dict = self.save_network_in_dictionary()
        with open( filename , 'wb') as afile:
            cPickle.dump( network_saved_dict, afile, 2 )


    # @staticmethod
    # def load_from_file( filename = "network.pkl" ):
    #     """
    #     Load the complete configuration of a previously stored network.
    #     """
    #     network = NeuralNet( 0, 0, 0, 0, [0,0], [0,0], [0,0])
    #
    #     with open( filename , 'rb') as afile:
    #         import cPickle
    #         store_dict = cPickle.load(afile)
    #
    #         n_inputs             = store_dict["n_inputs"]
    #         n_outputs            = store_dict["n_outputs"]
    #         n_hiddens            = store_dict["n_hiddens"]
    #         n_hidden_layers      = store_dict["n_hidden_layers"]
    #         neurons_ios = store_dict["neurons_ios"]
    #         weight_init_functions = store_dict["weight_init_functions"]
    #         learning_rate_functions = store_dict["learning_rate_functions"]
    #
    #     network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, neurons_ios, weight_init_functions, learning_rate_functions)
    #     network.set_links( store_dict["links"] )
    #     return network


    def __str__(self):
        return '\n'.join([str(i)+' '+str(layer) for i, layer in enumerate(self.layers)])


class NetworkDataCollector:
    def __init__(self, network, data_collection_interval=1):
        self.network = network
        self.data_dictionary = defaultdict(dict)
        self.data_collection_interval = data_collection_interval
        
    def store(self, epoch, example_number):
        if epoch%self.data_collection_interval == 0:
            self.data_dictionary[epoch][example_number] = deepcopy(self.network)  #store the network
            return
        if self.network.nearly_done:
            self.data_dictionary["end"][example_number] = deepcopy(self.network)  #store the network
       
    def extract_weights(self, layer_number=0, example_number=0):
        weights = []
        epoch_idx = []
        neuron_idx = []
        weight_idx = []
        keys = self.data_dictionary.keys()
        for key in keys:
            net_snapshot = self.data_dictionary[key][example_number]
            for i_neuron, neuron in enumerate(net_snapshot.layers[layer_number].learning_neurons):
                for i_weight, link in enumerate(neuron.links):
                    epoch_idx.append(key)
                    neuron_idx.append(i_neuron)
                    weight_idx.append(i_weight)
                    weights.append(link.weight)
        weight_series = Series(weights, index=[epoch_idx, neuron_idx, weight_idx])
        weight_series.index.names = ['epochs','neuron_num','weight_num']
        return weight_series

class Instance:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __str__(self):
        return 'Features: %s, Targets: %s' % ( str(self.features), str(self.targets) )

class Link:
    def __init__(self, weight_init_function, network):
        self.weight_init_function = weight_init_function
        self.network = network
        self.weight = self.weight_init_function()
        self.source_neuron = None

    def set_source_neuron(self, source_neuron):
        self.source_neuron = source_neuron

    def calc_links_output(self):
        return self.weight * self.source_neuron.output

    def adapt_links_weight(self, error_at_input, learning_rate_function):
        delta_w = learning_rate_function() * error_at_input * self.source_neuron.output
        self.weight += delta_w

    def __str__(self):
        return str(self.weight)

class LinkMemory(Link):
    def __init__(self, weight_init_function, network):
        Link.__init__(self, weight_init_function, network)
        self.input_queue = deque([0.5]*(network.n_delays))
        self.delayed_input = None

    def calc_links_output(self):
        self.input_queue.append(self.source_neuron.output)
        self.delayed_input = self.input_queue.popleft()
        return self.weight * self.delayed_input

    def adapt_links_weight(self, error_at_input, learning_rate_function):
        delta_w = learning_rate_function() * error_at_input * self.delayed_input
        self.weight += delta_w


class BiasNeuron:
    def __init__(self):
        self.output = 1.0
        self.output_links = []
        self.network = None

    def __str__(self):
        return 'BiasNeuron, Output = ' + str(self.output)

class InputNeuron:
    def __init__(self, neuron_id, network):
        self.neuron_id = neuron_id
        self.neuron_number = neuron_id[1]
        self.output = None
        self.output_links = []
        self.network = network

    def calc_neurons_output(self):
        inputs = self.network.inputs_for_training_example
        self.output = inputs[self.neuron_number]
        return self.output

    def __str__(self):
        return 'NeuronForInput, Output = ' + str(self.output)


class HiddenNeuron:
    def __init__(self, neuron_id, n_inputs, network, activation_function, weight_init_function, learning_rate_function):
        self.neuron_id = neuron_id
        self.neuron_number = neuron_id[1]
        self.n_inputs = n_inputs
        self.network = network
        self.links = []
        #self.links = [Link(weight_init_function, network) for _ in range(0, n_inputs + 1)]  # +1 for bias link
        self.output_links = []
        self.activation_function = activation_function
        self.weight_init_function = weight_init_function
        self.learning_rate_function = learning_rate_function
        self.error = None
        self.error_at_input = None
        self.netinput = None
        self.output = None

    def calc_neurons_output(self):
        self.netinput = 0.0
        for link in self.links:
           self.netinput += link.calc_links_output()
        self.output = self.activation_function.io( self.netinput )
        return self.output

    def calc_neurons_error(self, upper_layer):
        upper_layer_input_errors = [neuron.error_at_input for neuron in upper_layer.learning_neurons]
        self.error = np.dot(upper_layer_input_errors, [link.weight for link in self.output_links])
        self.calc_neurons_input_error()
        
    def calc_neurons_input_error(self):
        self.error_at_input  = self.error * self.activation_function.io_derivative( self.netinput, self.output)
        
    def adapt_weights_in_links(self):
        for link in self.links:
            link.adapt_links_weight(self.error_at_input, self.learning_rate_function)

    def __str__(self):
        weights = [link.weight for link in self.links]
        return 'Neuron Weights: %s  Net Input: %s  Output: %s' % (str(weights), str(self.netinput),  str(self.output))
 
 
class OutputNeuron(HiddenNeuron):
    def __init__(self, neuron_id, n_inputs, network, activation_function, weight_init_function, learning_rate_function):
        HiddenNeuron.__init__(self, neuron_id, n_inputs, network, activation_function, weight_init_function, learning_rate_function)
        self.output_links = None


class NeuronElman:
    def __init__(self, neuron_id, network, weight_init_function, learning_rate_function):
        HiddenNeuron.__init__(self, neuron_id, 1, network, LinearIO(), weight_init_function, learning_rate_function)

    def __str__(self):
        return 'NeuronElman, Output = ' + str(self.output)


class HiddenNeuronLayer:
    #def __init__(self, layer_number, n_neurons, n_inputs, network, activation_function, weight_init_function, learning_rate_function):
    def __init__(self, layer_number, network):
        self.layer_number = layer_number
        self.network = network
        n_neurons_for_each_layer = network.n_neurons_for_each_layer
        n_neurons = n_neurons_for_each_layer[layer_number]
        n_inputs = n_neurons_for_each_layer[layer_number-1]
        neurons_io = network.neurons_ios[layer_number]
        weight_init_function = network.weight_init_functions[layer_number]
        learning_rate_function = network.learning_rate_functions[layer_number]

        self.neurons = [HiddenNeuron((layer_number, neuron_number), n_inputs, network, neurons_io, weight_init_function, learning_rate_function) for neuron_number in range(0, n_neurons)]

        self.neurons.append(BiasNeuron())
        self.neurons_wo_bias_neuron = self.neurons[:-1]
        self.learning_neurons = self.neurons_wo_bias_neuron

    def __str__(self):
        return 'Hidden Layer:\n\t'+'\n\t'.join([str(neuron) for neuron in self.neurons])+''


class InputNeuronLayer:
    # This version does not currently contain 'learning elman neurons'
    def __init__(self, layer_number, network):
        self.network = network
        self.layer_number = layer_number
        self.n_neurons = network.n_neurons_for_each_layer[layer_number]
        self.neurons = [InputNeuron((layer_number, neuron_number), network) for neuron_number in range(0, self.n_neurons)]
        self.neurons.append(BiasNeuron())
        self.neurons_wo_bias_neuron = self.neurons[:-1]
        self.learning_neurons = []

    def __str__(self):
        return 'Layer of Inputs:\n\t'+'\n\t'.join([str(neuron) for neuron in self.neurons])+''


class OutputNeuronLayer:
    #def __init__(self, layer_number, n_neurons, n_inputs, network, activation_function, weight_init_function, learning_rate_function):
    def __init__(self, layer_number, network):
        self.layer_number = layer_number
        self.network = network
        n_neurons_for_each_layer = network.n_neurons_for_each_layer
        n_neurons = n_neurons_for_each_layer[layer_number]
        n_inputs = n_neurons_for_each_layer[layer_number-1]
        neurons_io = network.neurons_ios[layer_number]
        weight_init_function = network.weight_init_functions[layer_number]
        learning_rate_function = network.learning_rate_functions[layer_number]

        self.neurons = [OutputNeuron((layer_number, neuron_number), n_inputs, network, neurons_io, weight_init_function, learning_rate_function) for neuron_number in range(0, n_neurons)]
        self.neurons_wo_bias_neuron = self.neurons
        self.learning_neurons = self.neurons
        
    def __str__(self):
        return 'Layer for Output:\n\t'+'\n\t'.join([str(neuron) for neuron in self.neurons])+''



# NOTES:
"""

class NeuralNet:
 
    all_data = []
    
    @staticmethod  # This is meant to be a class method
    def save_delta_ws(delta_ws):
        NeuralNet.all_data += delta_ws
    
    @staticmethod  # This is meant to be a class method  
    def reset_all_delta_ws():
        NeuralNet.all_data = []

"""
#def post_process(data_collector):
    ## fig = plt.figure()
    ## ax1 = fig.add_subplot(2,1,1)
    #weight_series = data_collector.extract_weights(layer_number=0)
    
    #w0_series = weight_series[:,0,0]
    #w0_series.name = "weight0"
    #w1_series = weight_series[:,0,1]
    #w1_series.name = "weight1"
    
    #ratios = w0_series / w1_series
    #convert_to_angles_function = lambda x: 180.0 * math.atan(x) / math.pi
    #extracted_series = ratios.map(convert_to_angles_function)
    #extracted_series.name = "hyperplane_angle"

    #df = DataFrame([w0_series, w1_series, extracted_series])   
    #print df
    
    #sorted_extracted_series = extracted_series.sort_index()
    #sorted_extracted_series.plot()
    #plt.show()
