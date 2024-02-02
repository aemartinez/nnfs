# Neural Network from scratch

This is my implementation of a feedforward neural network with ReLU activation functions. I loosely followed the workflow outlined in the book https://nnfs.io.

## Project organization

The project has four main files:
- ``NeuralNetwork.py`` includes all the necessary classes for performing a forward and backward pass in the network.
- ``Optimizer.py`` contains the stochastic gradient descent optimizer implementation and the ``Trainer`` class responsible for training.
- ``Evaluation.py`` includes the class that calculates the model's ``Accuracy``.
- ``main.py`` holds the code that trains and tests the neural network on a classification problem using spiral data.

Requirements are specified in ``requirements.txt``.
