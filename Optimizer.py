import numpy as np
from NeuralNetwork import NeuralNetwork, Layer, ActivationReLU, ActivationSoftmax, LossCategoricalCrossentropy
from Evaluation import Accuracy

class OptimizerSGD :
    
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__ (self, learning_rate: float = 1.0, decay: float = 0.0, momentum: float = 0.0):
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.current_iteration = 0
    
    def update_learning_rate(self):
        self.current_learning_rate = self.initial_learning_rate * (1. / (1. + self.decay * self.current_iteration))

    # Update parameters
    def update_layer_params (self, layer: Layer ):
        if self.momentum:
            weights_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weights_updates
            biases_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = biases_updates
        else: 
            weights_updates = - self.current_learning_rate * layer.dweights
            biases_updates = - self.current_learning_rate * layer.dbiases

        layer.weights += weights_updates
        layer.biases += biases_updates

    def update_iteration(self):
        self.current_iteration += 1

    def update_network_params(self, network: NeuralNetwork):
        for layer in network.layers:
            self.update_layer_params(layer)

class Trainer:
    def __init__(self, optimizer: OptimizerSGD):
        self.optimizer = optimizer

    def set_training_data(self, data, true_labels):
        self.training_data = data.copy()
        self.training_true_labels = true_labels.copy()

    def set_validation_data(self, data, true_labels):
        self.validation_data = data.copy()
        self.validation_true_labels = true_labels.copy()

    def set_neural_network(self, network: NeuralNetwork):
        self.network = network

    def set_loss_function(self, loss_function: LossCategoricalCrossentropy):
        self.loss_function = loss_function

    def train(self, n_iterations):
            
        self.network.forward(self.training_data)
        loss = self.loss_function.calculate(self.network.output, self.training_true_labels)
        print(f"loss: {loss:.3f}")

        self.loss_function.backward(self.network.output, self.training_true_labels)
        dvalues = self.loss_function.dinputs
        for i in range(n_iterations):
            
            self.network.backward(dvalues)
            
            self.optimizer.update_learning_rate()
            self.optimizer.update_network_params(self.network)
            self.optimizer.update_iteration()
            
            if i % 100 == 0:
                self.print_performance(i)

            self.network.forward(self.training_data)
            self.loss_function.backward(self.network.output, self.training_true_labels)
            dvalues = self.loss_function.dinputs

        self.print_performance(n_iterations)

    def print_performance(self, iteration_number: int):

        accuracy_function = Accuracy()
        print()
        print("TRAINING DATA")
        self.network.forward(self.training_data)
        loss = self.loss_function.calculate(self.network.output, self.training_true_labels)
        accuracy = accuracy_function.calculate(self.network.output, self.training_true_labels)
        print(f"i: {iteration_number}, lr: {self.optimizer.current_learning_rate:.3f}, loss: {loss:.3f}, acc: {accuracy:.3f}.")

        print("VALIDATION DATA")
        self.network.forward(self.validation_data)
        loss = self.loss_function.calculate(self.network.output, self.validation_true_labels)
        accuracy = accuracy_function.calculate(self.network.output, self.validation_true_labels)
        print(f"i: {iteration_number}, lr: {self.optimizer.current_learning_rate:.3f}, loss: {loss:.3f}, acc: {accuracy:.3f}.")
