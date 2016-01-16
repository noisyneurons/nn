# /home/mark/anaconda/lib/python2.7/site-packages/_pytest/config.py
# run these test using pytest.py


import math
import random

import numpy as np
from jorg.neuralnet2 import Neuron, NeuralNet, Instance
from linear_algebra import dot

from old.activation_functions import sigmoid_function, linear_function


def weight_init_function_random(sequence_num=0):
    return random.uniform(-0.5,0.5)
    
def learning_rate_function(phase=0):
    return 1.0


class TestNeuron:
    def setup1(self):
        random.seed(1)
        aNeuron = Neuron(n_inputs=2, activation_function=sigmoid_function, weight_init_function=weight_init_function_random, learning_rate_function=learning_rate_function)
        aNeuron.weights = [1.0,1.0,1.0]
        inputs = [1.0,1.0,1.0]
        output = aNeuron.calc_neurons_output(inputs)
        expected = sigmoid_function( dot(inputs, aNeuron.weights) )
        return output, expected
    
    def test_answer(self):
        output, expected = self.setup1()
        diff = abs(output - expected)
        assert diff < 0.00001
        
##
    def setup2(self):
        random.seed(1)
        aNeuron = Neuron(n_inputs=2, activation_function=sigmoid_function, weight_init_function=weight_init_function_random, learning_rate_function=learning_rate_function)
        aNeuron.weights = [1.0,0.5,0.25]
        inputs = [2.0,1.5,1.0]
        output = aNeuron.calc_neurons_output(inputs)
        expected = sigmoid_function( dot(inputs, aNeuron.weights) )
        return output, expected
    
    def test_answer2(self):
        output, expected = self.setup1()
        diff = abs(output - expected)
        assert diff < 0.00001

##################################################################

class TestNeuralNet:
    def setup1(self):
        n_inputs = 1
        n_outputs = 1
        n_hiddens = 0
        n_hidden_layers = 0
        activation_functions = [ linear_function ]
        weight_init_functions = [ weight_init_function_random ]
        learning_rate_functions = [ learning_rate_function ] 
        network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions, weight_init_functions, learning_rate_functions)
        return network
  
    def test_0(self):
        aTrainingExample =  Instance( [2.0], [0] )
        network = self.setup1()
        network.layers[0].neurons[0].weights = [1.0, 0.0]
        outputs = network.calc_networks_output(aTrainingExample.features)
        result = outputs[0]
        expected = 2.0
        diff = abs(result - expected)
        assert diff < 0.00001

    def test_1(self):
        aTrainingExample =  Instance( [0.0], [0] )
        network = self.setup1()
        network.layers[0].neurons[0].weights = [1.0, 1.0]
        outputs = network.calc_networks_output(aTrainingExample.features)
        result = outputs[0]
        expected = 1.0
        diff = abs(result - expected)
        assert diff < 0.00001
        
    def test_2(self):
        aTrainingExample =  Instance( [1.0], [0] )
        network = self.setup1()
        network.layers[0].neurons[0].weights = [1.0, -1.0]
        outputs = network.calc_networks_output(aTrainingExample.features)
        result = outputs[0]
        expected = 0.0
        diff = abs(result - expected)
        assert diff < 0.00001

##################
    def setup2(self):
        n_inputs = 1
        n_outputs = 1
        n_hiddens = 0
        n_hidden_layers = 0
        activation_functions = [ sigmoid_function ]
        weight_init_functions = [ weight_init_function_random ]
        learning_rate_functions = [ learning_rate_function ] 
        network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions, weight_init_functions, learning_rate_functions)
        return network
  
    def test_3(self):
        aTrainingExample =  Instance( [1.0], [0] )
        network = self.setup2()
        network.layers[0].neurons[0].weights = [1.0, 0.0]
        outputs = network.calc_networks_output(aTrainingExample.features)
        result = outputs[0]
        expected = 1.0 / (1.0 + math.exp(-1.0))
        diff = abs(result - expected)
        assert diff < 0.00001

##################
    def setup_single_hidden_layer_net(self):
        n_inputs = 1
        n_outputs = 1
        n_hiddens = 1
        n_hidden_layers = 1
        activation_functions = [ linear_function, linear_function ]
        weight_init_functions = [ weight_init_function_random, weight_init_function_random ]
        learning_rate_functions = [ learning_rate_function, learning_rate_function ] 
        network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions, weight_init_functions, learning_rate_functions)
        return network
   
    def test_hidden_layer_net0(self):
        aTrainingExample =  Instance( [2.5], [0] )
        network = self.setup_single_hidden_layer_net()
        network.layers[0].neurons[0].weights = [1.0, 0.0]
        network.layers[1].neurons[0].weights = [1.0, 0.0]
        outputs = network.calc_networks_output(aTrainingExample.features)
        result = outputs[0]
        expected = 2.5
        diff = abs(result - expected)
        assert diff < 0.00001
        
        aTrainingExample =  Instance( [2.5], [1.0] )
        network.layers[0].neurons[0].weights = [1.0, 1.0]
        outputs = network.calc_networks_output(aTrainingExample.features)
        result = outputs[0]
        expected = 3.5
        diff = abs(result - expected)
        assert diff < 0.00001
        
##################
    def setup_single_hidden_layer_net1(self):
        n_inputs = 1
        n_outputs = 1
        n_hiddens = 1
        n_hidden_layers = 1
        activation_functions = [ sigmoid_function, linear_function ]
        weight_init_functions = [ weight_init_function_random, weight_init_function_random ]
        learning_rate_functions = [ learning_rate_function, learning_rate_function ] 
        network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions, weight_init_functions, learning_rate_functions)
        return network
   
    def test_hidden_layer_net1(self):
        aTrainingExample =  Instance( [2.0], [1.0] )
        network = self.setup_single_hidden_layer_net1()
        network.layers[0].neurons[0].weights = [1.0, 1.0]
        network.layers[1].neurons[0].weights = [1.0, 0.0]
        network.activation_functions = [ sigmoid_function, linear_function ]
        outputs = network.calc_networks_output(aTrainingExample.features)
        result = outputs[0]
        expected = 1.0 / (1.0 + math.exp(-3.0))
        diff = abs(result - expected)
        assert diff < 0.00001
##################

    def setup_single_hidden_layer_net2(self):
        n_inputs = 1
        n_outputs = 1
        n_hiddens = 1
        n_hidden_layers = 1
        activation_functions = [ sigmoid_function, sigmoid_function ]
        weight_init_functions = [ weight_init_function_random, weight_init_function_random ]
        learning_rate_functions = [ learning_rate_function, learning_rate_function ] 
        network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions, weight_init_functions, learning_rate_functions)
        return network
   
    def test_hidden_layer_net2(self):
        aTrainingExample =  Instance( [0.5], [0.0] )
        network = self.setup_single_hidden_layer_net2()
        network.layers[0].neurons[0].weights = [1.0, 1.0]
        network.layers[1].neurons[0].weights = [1.0, 1.0]
        
        outputs = network.calc_networks_output(aTrainingExample.features)
        
        result = network.layers[0].neurons[0].output
        expected_hidden_layer_output = 1.0 / (1.0 + math.exp(-1.5))
        diff = abs(result - expected_hidden_layer_output)
        assert diff < 0.00001
        
        sum_of_weighted_inputs_to_output_neuron = expected_hidden_layer_output  + 1.0
        expected_output_neuron_output = 1.0 / (1.0 + math.exp(-1.0 * sum_of_weighted_inputs_to_output_neuron ))
        result = outputs[0]
        diff = abs(result - expected_output_neuron_output)
        assert diff < 0.00001

######################################################
 
class TestNeuralNetLearning:
    
    def weight_init_function_test1(self,i):
        weights = [2.0, 3.0]
        return weights[i]

    def weight_init_function_test2(self,i):
        weights = [2.0, 1.0]
        return weights[i]

    def setup1(self, activation_functions, weight_init_functions, learning_rate_functions):
        n_inputs = 1
        n_outputs = 1
        n_hiddens = 0
        n_hidden_layers = 0
        network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions, weight_init_functions, learning_rate_functions)
        return network
        
    def test_calc_output_neurons_errors(self):
        activation_functions = [ linear_function ]
        weight_init_functions = [ weight_init_function_random ]
        learning_rate_functions = [ learning_rate_function ]
        network = self.setup1(activation_functions, weight_init_functions, learning_rate_functions)
        errors = network.calc_output_neurons_errors(np.array([0.0]), np.array([0.5]))
        assert errors==[-0.5]
                
    def test_calc_output_neurons_errors2(self):
        activation_functions = [ linear_function ]
        weight_init_functions = [ self.weight_init_function_test1 ]
        learning_rate_functions = [ learning_rate_function ]
        network = self.setup1(activation_functions, weight_init_functions, learning_rate_functions)
        network_outputs = network.calc_networks_output( [1.0, 1.0] )        
        assert network.layers[-1].neurons[0].weights[0]==2.0 
        assert network.layers[-1].neurons[0].weights[1]==3.0 
        
        assert network_outputs == [5.0]
        
        errors = network.calc_output_neurons_errors(network_outputs, [5.5])
        assert errors==[-0.5]
        output_neurons_error = network.layers[-1].neurons[0].error
        assert output_neurons_error == -0.5
        

    def test_calc_hidden_neurons_errors1(self):
        activation_functions = [ linear_function ] 
        weight_init_functions = [ self.weight_init_function_test1 ]
        learning_rate_functions = [ learning_rate_function ]        
        network = self.setup1(activation_functions, weight_init_functions, learning_rate_functions)
        network.layers[-1].neurons[0].weights = 2.0, 3.0
        network_outputs = network.calc_networks_output( [1.0, 1.0] )
        network.calc_output_neurons_errors(network_outputs, [5.5])

        error_at_input_of_the_output_neuron = network.layers[-1].neurons[0].error_at_input
        assert error_at_input_of_the_output_neuron == -0.5
        
    def setup_single_hidden_layer_net2(self, activation_functions, weight_init_functions, learning_rate_functions):
        n_inputs = 1
        n_outputs = 1
        n_hiddens = 1
        n_hidden_layers = 1
        return NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions, weight_init_functions, learning_rate_functions)
   
    def test_calc_hidden_neurons_errors2(self): 
        activation_functions = [ linear_function, linear_function ]     
        weight_init_functions = [ self.weight_init_function_test2, self.weight_init_function_test2 ]
        learning_rate_functions = [ learning_rate_function, learning_rate_function] 
        network = self.setup_single_hidden_layer_net2(activation_functions, weight_init_functions, learning_rate_functions)
        
        network_outputs = network.calc_networks_output( [1.0, 1.0] )
        hidden_neuron = network.layers[0].neurons[0]    
        assert hidden_neuron.netinput == 3.0
        output_neuron = network.layers[1].neurons[0]
        assert output_neuron.netinput == 7.0
        assert network_outputs[0] == 7.0
        
        network.calc_output_neurons_errors(network_outputs, [5.5])
        assert output_neuron.error == 1.5
        assert output_neuron.error_at_input == 1.5
        
        network.calc_hidden_neurons_errors()
        assert hidden_neuron.error == 3.0
        assert hidden_neuron.error_at_input == 3.0
        
    def test_calc_hidden_neurons_errors3(self): 
        activation_functions = [ sigmoid_function, linear_function ]     
        weight_init_functions = [ self.weight_init_function_test2, self.weight_init_function_test2 ]
        learning_rate_functions = [ learning_rate_function, learning_rate_function] 
        network = self.setup_single_hidden_layer_net2(activation_functions, weight_init_functions, learning_rate_functions)
        
        network_outputs = network.calc_networks_output( [1.0, 1.0] )
        hidden_neuron = network.layers[0].neurons[0]    
        assert hidden_neuron.netinput == 3.0
        expected_hiddens_output = sigmoid_function(3.0)
        assert hidden_neuron.output == sigmoid_function(3.0)
        output_neuron = network.layers[1].neurons[0]
        expected_netinput_to_output_neuron = (2.0 * expected_hiddens_output) + 1.0
        assert output_neuron.netinput == expected_netinput_to_output_neuron
        assert network_outputs[0] == expected_netinput_to_output_neuron
        
        network.calc_output_neurons_errors(network_outputs, [5.5])
        expected_output_error = expected_netinput_to_output_neuron - 5.5
        assert output_neuron.error == expected_output_error
        assert output_neuron.error_at_input == expected_output_error
 
    def test_calc_hidden_neurons_errors4(self): 
        activation_functions = [ sigmoid_function, sigmoid_function ]     
        weight_init_functions = [ self.weight_init_function_test2, self.weight_init_function_test2 ]
        learning_rate_functions = [ learning_rate_function, learning_rate_function ]
        network = self.setup_single_hidden_layer_net2(activation_functions, weight_init_functions, learning_rate_functions)
        
        network_outputs = network.calc_networks_output( [1.0, 1.0] )
        hidden_neuron = network.layers[0].neurons[0]    
        assert hidden_neuron.netinput == 3.0
        expected_hiddens_output = sigmoid_function(3.0)
        assert hidden_neuron.output == sigmoid_function(3.0)
        output_neuron = network.layers[1].neurons[0]
        expected_netinput_to_output_neuron = (2.0 * expected_hiddens_output) + 1.0
        assert output_neuron.netinput == expected_netinput_to_output_neuron
        assert network_outputs[0] == sigmoid_function(expected_netinput_to_output_neuron)
        
        network.calc_output_neurons_errors(network_outputs, [5.5])
        expected_output_error = sigmoid_function(expected_netinput_to_output_neuron) - 5.5
        assert output_neuron.error == expected_output_error
        assert output_neuron.error_at_input == (expected_output_error * sigmoid_function( expected_netinput_to_output_neuron, True ))
  
######################################################

class TestNetWeightChanges:
    
    def weight_init_function_test1(self,i):
        weights = [2.0, 3.0]
        return weights[i]

    def weight_init_function_test2(self,i):
        weights = [2.0, 1.0]
        return weights[i]

    def setup1(self, activation_functions, weight_init_functions, learning_rate_functions):
        n_inputs = 1
        n_outputs = 1
        n_hiddens = 0
        n_hidden_layers = 0
        network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions, weight_init_functions, learning_rate_functions)
        return network
               
    def test_change_weights_for_example0a(self):
        activation_functions = [ linear_function ]
        weight_init_functions = [ self.weight_init_function_test1 ]
        learning_rate_functions = [ learning_rate_function ]
        network = self.setup1(activation_functions, weight_init_functions, learning_rate_functions)
        single_training_example_inputs = [1.0, 1.0]  # includes Bias!
        network_outputs = network.calc_networks_output( single_training_example_inputs )        

        assert network_outputs == [5.0]
        
        errors = network.calc_output_neurons_errors(network_outputs, [5.5])
        assert errors==[-0.5]
        output_neurons_error = network.layers[-1].neurons[0].error
        assert output_neurons_error == -0.5
        
        print network.layers[0]
                
        network.change_weights_for_example( single_training_example_inputs )
        assert network.layers[0].neurons[0].weights[0] == 1.5
        assert network.layers[0].neurons[0].weights[1] == 2.5
        
    def test_change_weights_for_example0b(self):
        activation_functions = [ sigmoid_function ]
        weight_init_functions = [ self.weight_init_function_test1 ]
        learning_rate_functions = [learning_rate_function]
        network = self.setup1(activation_functions, weight_init_functions, learning_rate_functions)
        single_training_example_inputs = [1.0, 1.0]  # includes Bias!
        network_outputs = network.calc_networks_output( single_training_example_inputs )        
        
        expected_network_output = sigmoid_function(5.0)
        assert network_outputs == [expected_network_output]
        
        errors = network.calc_output_neurons_errors(network_outputs, [5.5])
        expected_output_error = expected_network_output - 5.5
        assert errors == [expected_output_error]
        output_neurons_error = network.layers[-1].neurons[0].error
        assert output_neurons_error == expected_output_error
        
        output_neurons_error_at_input = network.layers[-1].neurons[0].error_at_input
        output_neurons_netinput = network.layers[-1].neurons[0].netinput
        assert output_neurons_error_at_input == output_neurons_error * sigmoid_function(output_neurons_netinput, True)
        
        print network.layers[0]
                      
        network.change_weights_for_example( single_training_example_inputs )
        assert network.layers[0].neurons[0].weights[0] == 2.0 + (output_neurons_error_at_input * 1.0)
        assert network.layers[0].neurons[0].weights[1] == 3.0 + (output_neurons_error_at_input * 1.0)
        

    def test_change_weights_for_example1(self):
        activation_functions = [ linear_function ]
        weight_init_functions = [ self.weight_init_function_test2 ]
        learning_rate_functions = [learning_rate_function]
        network = self.setup1(activation_functions, weight_init_functions, learning_rate_functions)
        single_training_example_inputs = [1.0, 1.0]  # includes Bias!
        network_outputs = network.calc_networks_output( single_training_example_inputs )        

        assert network_outputs == [3.0]
        
        errors = network.calc_output_neurons_errors(network_outputs, [2.5])
        assert errors==[0.5]
        output_neurons_error = network.layers[-1].neurons[0].error
        assert output_neurons_error == 0.5
        
        network.calc_hidden_neurons_errors()
        
        print network.layers[0]
                
        network.change_weights_for_example( single_training_example_inputs )
        assert network.layers[0].neurons[0].weights[0] == 2.5
        assert network.layers[0].neurons[0].weights[1] == 1.5

    def setup_single_hidden_layer_net2(self, activation_functions, weight_init_functions, learning_rate_functions):
        n_inputs = 1
        n_outputs = 1
        n_hiddens = 1
        n_hidden_layers = 1
        return NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions, weight_init_functions, learning_rate_functions)
       
    def test_change_weights_for_example2(self):
        activation_functions = [ linear_function, linear_function ]
        weight_init_functions = [ self.weight_init_function_test2, self.weight_init_function_test2 ]
        learning_rate_functions = [learning_rate_function, learning_rate_function]
        network = self.setup_single_hidden_layer_net2(activation_functions, weight_init_functions, learning_rate_functions )
        single_training_example_inputs = [1.0, 1.0]  # includes Bias!
        network_outputs = network.calc_networks_output( single_training_example_inputs )        

        assert network_outputs == [7.0]
        
        errors = network.calc_output_neurons_errors(network_outputs, [6.0])
        assert errors==[1.0]
        output_neurons_error = network.layers[-1].neurons[0].error
        assert output_neurons_error == 1.0
        
        print network.layers[0]
        
        network.calc_hidden_neurons_errors()
        network.change_weights_for_example( single_training_example_inputs )
        
        assert network.layers[1].neurons[0].weights[0] == 5.0
        assert network.layers[1].neurons[0].weights[1] == 2.0
        
        assert network.layers[0].neurons[0].error == 2.0
        assert network.layers[0].neurons[0].error_at_input == 2.0
 
        assert network.layers[0].neurons[0].weights[0] == 4.0
        assert network.layers[0].neurons[0].weights[1] == 3.0
        
######################################################

class TestOneHiddenLayerLearning:
    
    def learning_rate_function(self, phase=0):
        return -0.5

    def setup0(self):
        random.seed(10)
        
        n_inputs = 2
        n_outputs = 1
        n_hiddens = 0
        n_hidden_layers = 0
        
        # specify activation functions per layer eg: [ hidden_layer_1, hidden_layer_2, output_layer ]
        activation_functions = [ sigmoid_function ]*n_hidden_layers + [ sigmoid_function ]
        weight_init_functions = [ weight_init_function_random ]*n_hidden_layers + [ weight_init_function_random ]
        learning_rate_functions = [ self.learning_rate_function ]*n_hidden_layers + [ self.learning_rate_function ]
        
        # initialize the neural network
        network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions, weight_init_functions, learning_rate_functions)
        return network
       
    def test_single_layer_net_on_OR(self):
        
        network = self.setup0()
        training_set =  [ Instance( [0.0,0.0], [0.0] ), Instance( [0.0,1.0], [1.0] ), Instance( [1.0,0.0], [1.0] ), Instance( [1.0,1.0], [1.0] ) ]
        # train
        epoch, MSE = network.backpropagation(training_set, ERROR_LIMIT=1e-2)
        
        assert epoch==340
        assert MSE==0.0099759064076452887
        
    
    def setup1(self):
        random.seed(10)
        
        n_inputs = 2
        n_outputs = 1
        n_hiddens = 3
        n_hidden_layers = 1
        
        # specify activation functions per layer eg: [ hidden_layer_1, hidden_layer_2, output_layer ]
        activation_functions = [ sigmoid_function ]*n_hidden_layers + [ sigmoid_function ]
        weight_init_functions = [ weight_init_function_random ]*n_hidden_layers + [ weight_init_function_random ]
        learning_rate_functions = [ self.learning_rate_function ]*n_hidden_layers + [ self.learning_rate_function ]
        
        # initialize the neural network
        network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions, weight_init_functions, learning_rate_functions)
        return network
       
        
    def test_hidden_layer_net_on_OR(self):
        
        network = self.setup1()
        training_set =  [ Instance( [0.0,0.0], [0.0] ), Instance( [0.0,1.0], [1.0] ), Instance( [1.0,0.0], [1.0] ), Instance( [1.0,1.0], [1.0] ) ]
        # train
        epoch, MSE = network.backpropagation(training_set, ERROR_LIMIT=1e-2)
        
        assert epoch==758
        assert MSE==0.0099973561968336455
        
    def test_hidden_layer_net_on_XOR(self):
        
        network = self.setup1()
        training_set =  [ Instance( [0.0,0.0], [0.0] ), Instance( [0.0,1.0], [1.0] ), Instance( [1.0,0.0], [1.0] ), Instance( [1.0,1.0], [0.0] ) ]
        # train
        epoch, MSE = network.backpropagation(training_set, ERROR_LIMIT=1e-2)
        
        assert epoch==4098
        assert MSE==0.0099993587735512333

