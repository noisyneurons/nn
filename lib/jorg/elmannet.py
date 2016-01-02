# elmannet.py


# %matplotlib inline
from __future__ import division
import math
import random
from collections import defaultdict
from copy import deepcopy

from jorg.activation_classes import LinearIO

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def weight_init_function_random():
    return random.uniform(-0.5,0.5)
    
def learning_rate_function():
    return -0.1        
    
def intermediate_post_process(trial_params, data_collector, dfs_concatenated):
    weight_series = data_collector.extract_weights(layer_number=0)
    
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
            for i_neuron, neuron in enumerate(net_snapshot.layers[layer_number].neurons_wo_bias_neuron):
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
        self.features = features + [1.0]  # the appended [1.0] is the "bias input"
        self.targets = targets
        
    def __str__(self):
        return 'Features: %s, Targets: %s' % ( str(self.features), str(self.targets) )

class Link:
    def __init__(self, weight_init_function):
        self.weight_init_function = weight_init_function
        self.weight = self.weight_init_function()
    
    def increment_weight_by(self, delta_w):
        self.weight += delta_w
    
    def __str__(self):
        return str(self.weight)


class Neuron:
    def __init__(self, neuron_id, n_inputs, network, activation_function, weight_init_function, learning_rate_function):
        self.neuron_id = neuron_id
        self.n_inputs = n_inputs
        self.network = network
        self.links = [ Link(weight_init_function) for _ in range(0, n_inputs+1)]  # +1 for bias link
        self.output_links = []
        self.activation_function = activation_function
        self.weight_init_function = weight_init_function
        self.learning_rate_function = learning_rate_function
        self.error = None
        self.error_at_input = None
        self.netinput = None
        self.output = None

    def calc_neurons_output(self, inputs ):
        self.netinput = np.dot(inputs, [link.weight for link in self.links])
        self.output = self.activation_function.io( self.netinput )
        return self.output
        
    def calc_neurons_error(self, upper_layer):
        errors_at_next_layer = [neuron.error_at_input for neuron in upper_layer.neurons_wo_bias_neuron]
        self.error = np.dot(errors_at_next_layer, [link.weight for link in self.output_links])
        self.calc_neurons_input_error()
        
    def calc_neurons_input_error(self):
        self.error_at_input  = self.error * self.activation_function.io_derivative( self.netinput, self.output)
        
    def change_links(self, inputs_to_layer):
        delta_ws = [ (self.learning_rate_function() * self.error_at_input * neurons_input) for neurons_input in inputs_to_layer] 
        for link, delta_w in zip(self.links, delta_ws):
            link.increment_weight_by(delta_w)
 
    def __str__(self):
        weights = [link.weight for link in self.links]
        return 'Neuron Weights: %s  Net Input: %s  Output: %s' % (str(weights), str(self.netinput),  str(self.output))
 
 
class OutputNeuron(Neuron):
    def __init__(self, neuron_number, n_inputs, network, activation_function, weight_init_function, learning_rate_function):
        Neuron.__init__(self, neuron_number, n_inputs, network, activation_function, weight_init_function, learning_rate_function )


class BiasNeuron:
    def __init__(self):
        self.output = 1.0
        self.output_links = []
        self.network = None
    
    def calc_neurons_output(self, _ ):
        return self.output
                        
    def __str__(self):
        return 'BiasNeuron, Output = ' + str(self.output)
        
class InputNeuron:
    def __init__(self):
        BiasNeuron.__init__(self)

    def __str__(self):
        return 'InputNeuron, Output = ' + str(self.output)

class ElmanNeuron:
    def __init__(self, neuron_number, network, weight_init_function, learning_rate_function):
        Neuron.__init__(self, neuron_number, 1, network, LinearIO(), weight_init_function, learning_rate_function )

    def __str__(self):
        return 'ElmanNeuron, Output = ' + str(self.output)



class NeuronLayer:
    def __init__(self, layer_number, n_neurons, n_inputs, network, activation_function, weight_init_function, learning_rate_function):
        self.layer_number = layer_number
        self.n_neurons = n_neurons
        self.neurons = [Neuron( (layer_number,neuron_number), n_inputs, network, activation_function, weight_init_function, learning_rate_function) for neuron_number in range(0, self.n_neurons)]
        self.neurons.append(BiasNeuron())
        self.neurons_wo_bias_neuron = self.neurons[:-1]
        self.network = network
        
    def connect_to_next_layer(self, upper_layer):
        for i, lower_neuron in enumerate(self.neurons):
            for upper_neuron in upper_layer.neurons_wo_bias_neuron:
                lower_neuron.output_links.append(upper_neuron.links[i])

    def __str__(self):
        return 'Layer:\n\t'+'\n\t'.join([str(neuron) for neuron in self.neurons])+''


class OutputNeuronLayer:
    def __init__(self, layer_number, n_neurons, n_inputs, network, activation_function, weight_init_function, learning_rate_function):
        self.layer_number = layer_number
        self.n_neurons = n_neurons
        self.neurons = [ OutputNeuron( (layer_number,neuron_number), n_inputs, network, activation_function, weight_init_function, learning_rate_function) for neuron_number in range(0, n_neurons)]
        self.neurons_wo_bias_neuron = self.neurons
        self.network = None
        
    def __str__(self):
        return 'Layer for Output:\n\t'+'\n\t'.join([str(neuron) for neuron in self.neurons])+''


class NeuralNet:
    
    def __init__(self, n_inputs, n_outputs, n_hiddens, n_hidden_layers, neurons_ios,
    weight_init_functions, learning_rate_functions):
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        
        self.layers = None
        self.upper_layers = None
        self.lower_layers = None
        self.upper_layers_in_reverse_order =  None
        self.lower_layers_in_reverse_order =  None 
        
        self.neurons_ios = neurons_ios
        self.weight_init_functions = weight_init_functions
        self.learning_rate_functions = learning_rate_functions
                
        self.epoch = None
        self.example_number = None
        self.not_done = True
        self.nearly_done = False

        # Do not touch
        self._create_network()
        self._n_links = None
        # end
    
    def attach_network_to(self, an_object):
        an_object.network = self
    
    def _create_network(self):
        
        if self.n_hidden_layers == 0:
            # If we don't require hidden layers, only create output layer
            layer_number=0
            self.layers = [ OutputNeuronLayer( layer_number, self.n_outputs, self.n_inputs, self, self.neurons_ios[0], self.weight_init_functions[0], self.learning_rate_functions[0] )]
            
        else:
            # create the first hidden layer
            layer_number=0
            self.layers = [NeuronLayer( layer_number, self.n_hiddens, self.n_inputs, self, self.neurons_ios[0], self.weight_init_functions[0], self.learning_rate_functions[0] )]

            # create remaining hidden layers
            self.layers += [NeuronLayer( layer_number, self.n_hiddens, self.n_hiddens, self, self.neurons_ios[i+1], self.weight_init_functions[i+1], self.learning_rate_functions[i+1] ) for layer_number in range(1, (self.n_hidden_layers) )]

            # create output layer
            layer_number = self.n_hidden_layers
            self.layers += [OutputNeuronLayer( layer_number, self.n_outputs, self.n_hiddens, self, self.neurons_ios[self.n_hidden_layers], self.weight_init_functions[self.n_hidden_layers], self.learning_rate_functions[self.n_hidden_layers]  )]
               
            self.upper_layers = [aLayer for aLayer in self.layers[1:] ] 
            self.lower_layers = [aLayer for aLayer in self.layers[:-1]] 
            
             # add references to "output_links" going to next layer
            for lower_layer, upper_layer in zip(self.lower_layers, self.upper_layers):
                lower_layer.connect_to_next_layer(upper_layer)

            # Make subsequent backward error propagation code easier to write/understand!
            self.upper_layers_in_reverse_order =  [aLayer for aLayer in reversed(self.layers[1:])] 
            self.lower_layers_in_reverse_order =  [aLayer for aLayer in reversed(self.layers[:-1])] 
            
    def get_links(self):
        links = []
        for layer in self.layers:
            for neuron in layer.neurons_wo_bias_neuron:
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
            for neuron in layer.neurons_wo_bias_neuron:
                start, stop = stop, (stop + len(neuron.links))
                neuron.links = links[start:stop] 
        return self
 
    def calc_networks_output(self, inputs ):
        for layer in self.layers:
            outputs = []
            for neuron in layer.neurons:
                neurons_output = neuron.calc_neurons_output(inputs)
                outputs.append( neurons_output )
            inputs = outputs   
        return outputs
        
    def calc_output_neurons_errors(self, network_outputs, training_targets):
        # determine error at network's output
        errors = [ (aNetOutput - aTrainingTarget) for aNetOutput, aTrainingTarget in zip(network_outputs, training_targets) ]              
        # store output errors in output neurons 
        output_neurons = self.layers[-1].neurons
        for output_neuron, error in zip(output_neurons, errors):
            output_neuron.error = error
            output_neuron.calc_neurons_input_error()
        return errors
        
    def calc_hidden_neurons_errors(self):
        for lower_layer, upper_layer in zip(self.lower_layers_in_reverse_order, self.upper_layers_in_reverse_order):     
            for neuron in lower_layer.neurons_wo_bias_neuron:   # don't need to calculate bias neuron's error!!
                neuron.calc_neurons_error(upper_layer)

    def change_first_layer_links(self, inputs_to_layer):
        # The first hidden layer receives its inputs externally, not from lower layer neurons
        for neuron in self.layers[0].neurons_wo_bias_neuron:
            neuron.change_links(inputs_to_layer)

    def change_upper_layers_links(self):        
        for lower_layer, upper_layer in zip(self.layers[:-1], self.layers[1:]):
            lower_layer_neuron_outputs = [neuron.output for neuron in lower_layer.neurons]
            for neuron in upper_layer.neurons_wo_bias_neuron:
                neuron.change_links(lower_layer_neuron_outputs)

    def backpropagation(self, trainingset, ERROR_LIMIT, data_collector): 
        n_training_examples = len(trainingset)
        training_inputs  =  [instance.features for instance in trainingset]
        training_targets =  [instance.targets for instance in trainingset]
        
        MSE      = 1000.0 # any large enough number is good enough!      
        self.epoch = 0
        self.not_done = True
        self.nearly_done = False
        
        # epoch loop
        while self.not_done:
        
            if self.epoch > 300000:
                self.nearly_done = True
                return self.epoch, MSE
                
            collected_errors = []
            
            # 1-training-example per iteration
            for example_number, training_example_inputs in enumerate(training_inputs):
                
                self.example_number = example_number
                
                network_outputs = self.calc_networks_output( training_example_inputs )
                
                collected_errors += self.calc_output_neurons_errors(network_outputs, training_targets[example_number])
                
                if self.n_hidden_layers > 0:
                    self.calc_hidden_neurons_errors()
                
                self.change_first_layer_links(training_example_inputs)
                
                if self.n_hidden_layers > 0:   
                    self.change_upper_layers_links()

                data_collector.store(self.epoch, example_number)                       
            
            sse = np.dot(collected_errors, collected_errors)
            MSE = sse / (self.n_outputs * n_training_examples)
            if self.nearly_done:
                self.not_done = False
            if MSE < ERROR_LIMIT:
                self.nearly_done = True
            self.epoch += 1
        
        return self.epoch, MSE
            
    def save_network_in_dictionary(self):
        links = self.get_links()  # need this to be called before "self.no_links" -- need to CLEAN this smell!
        return {
            "n_inputs"              : self.n_inputs,
            "n_outputs"             : self.n_outputs,
            "n_hiddens"             : self.n_hiddens,
            "n_hidden_layers"       : self.n_hidden_layers,
            "neurons_ios"           : self.neurons_ios,
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
            
    
    @staticmethod
    def load_from_file( filename = "network.pkl" ):
        """
        Load the complete configuration of a previously stored network.
        """
        network = NeuralNet( 0, 0, 0, 0, [0], [0], [0])
        
        with open( filename , 'rb') as afile:
            import cPickle
            store_dict = cPickle.load(afile)
            
            n_inputs             = store_dict["n_inputs"]            
            n_outputs            = store_dict["n_outputs"]           
            n_hiddens            = store_dict["n_hiddens"]           
            n_hidden_layers      = store_dict["n_hidden_layers"]     
            neurons_ios = store_dict["neurons_ios"]
            weight_init_functions = store_dict["weight_init_functions"]
            learning_rate_functions = store_dict["learning_rate_functions"]
            
        network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, neurons_ios, weight_init_functions, learning_rate_functions)
        network.set_links( store_dict["links"] )  
        return network
                  
    def __str__(self):
        return '\n'.join([str(i+1)+' '+str(layer) for i, layer in enumerate(self.layers)])



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
