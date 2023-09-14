from nnfs.datasets import spiral_data, vertical_data
import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import NeuralNetwork, LossCategoricalCrossentropy
from Evaluation import Accuracy

X, y = spiral_data(samples=100, classes=3)

# X, y = vertical_data( samples = 100 , classes = 3 )
plt.scatter(X[:, 0 ], X[:, 1 ], c = y, s = 40 , cmap = 'brg' )
plt.show()

neural_network = NeuralNetwork(2, 1, 3, 3)

neural_network.forward(X)
output = neural_network.output

loss_function = LossCategoricalCrossentropy()
loss = loss_function.calculate(output, y)
print("loss before train: ", loss)

accuracy_function = Accuracy()
accuracy = accuracy_function.calculate(output, y)
print("accuracy: ", accuracy)

predictions = np.argmax(output, axis = 1)
plt.scatter(X[:, 0 ], X[:, 1 ], c = predictions, s = 40 , cmap = 'brg' )
plt.show()

neural_network.random_train(X, y, 1000)
neural_network.forward(X)
output = neural_network.output

loss = loss_function.calculate(output, y)
print("loss after train: ", loss)

accuracy_function = Accuracy()
accuracy = accuracy_function.calculate(output, y)
print("accuracy: ", accuracy)

predictions = np.argmax(output, axis = 1)
plt.scatter(X[:, 0 ], X[:, 1 ], c = predictions, s = 40 , cmap = 'brg' )
plt.show()