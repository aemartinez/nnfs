import numpy as np
import copy
from Evaluation import Accuracy

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, next_layer_dvalues):
        raise NotImplementedError
        self.dinputs = np.dot(next_layer_dvalues, self.weights.T)
        self.dweights = np.dot(self.inputs.T, next_layer_dvalues)
        self.dbiases = np.sum(next_layer_dvalues, axis = 0, keepdims = True)

class ActivationReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, next_layer_dvalues):
        self.dinputs = next_layer_dvalues.copy()
        self.dinputs[self.output <= 0] = 0

class ActivationSoftmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probabilities

    def backward(self, next_layer_dvalues):

        self.dinputs = np.empty_like(next_layer_dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, next_layer_dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class NeuralNetwork:

    def __init__(self, n_inputs, n_hidden_layers, layer_width, n_outputs):

        self.n_inputs = n_inputs
        self.n_hidden_layers = n_hidden_layers
        self.layer_width = layer_width
        self.n_outputs = n_outputs
        
        self.initialize_layers()

    def initialize_layers(self):
    
        self.layers = []

        layer_1 = Layer(self.n_inputs, self.layer_width)
        self.layers.append(layer_1)

        for i in range(self.n_hidden_layers - 1):
            new_layer = Layer(self.layer_width, self.layer_width)
            self.layers.append(new_layer)

        output_layer = Layer(self.layer_width, self.n_outputs)
        self.layers.append(output_layer)

    def random_perturbation(self):

        for i in range(len(self.layers)):
            self.layers[i].weights += 0.05 * np.random.randn(*self.layers[i].weights.shape)
            self.layers[i].biases += 0.05 * np.random.randn(*self.layers[i].biases.shape)

    def forward(self, inputs):

        self.inputs = inputs

        activation_relu = ActivationReLU()
        partial_output = inputs
        for i in range(len(self.layers) - 1):
            self.layers[i].forward(partial_output)
            partial_output = self.layers[i].output
            activation_relu.forward(partial_output)
            partial_output = activation_relu.output

        activation_softmax = ActivationSoftmax()
        self.layers[-1].forward(partial_output)
        partial_output = self.layers[-1].output
        activation_softmax.forward(partial_output)
        final_output = activation_softmax.output

        self.output = final_output

    def random_train(self, data, true_labels, n_iterations):

        loss_function = LossCategoricalCrossentropy()
        self.forward(data)
        best_loss = loss_function.calculate(self.output, true_labels)

        for i in range(n_iterations):

            old_layers = copy.deepcopy(self.layers)
            self.random_perturbation()

            self.forward(data)
            loss = loss_function.calculate(self.output, true_labels)
            if loss < best_loss:
                best_loss = loss
                print("new loss: ", best_loss)
            else:
                self.layers = old_layers

class Loss:
    
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class LossCategoricalCrossentropy(Loss):
    
    def forward(self, y_pred, y_true):
        number_samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1 - 1e-7 )

        correct_confidences = []

        #if y_true are categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(number_samples), 
                y_true 
            ]
        
        #if y_true are hot-one encoded labels
        if len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_true * y_pred_clipped,
                axis = 1
            )

        #calculate losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, next_layer_dvalues, y_true):
        
        number_samples = len(next_layer_dvalues)
        number_labels = len(next_layer_dvalues[0])
        
        # If labels are sparse, turn them into one-hot vector
        if len (y_true.shape) == 1 :
            y_true = np.eye(number_labels)[y_true]

        # Calculate gradient
        self.dinputs = - y_true / next_layer_dvalues
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples