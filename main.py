from nnfs.datasets import spiral_data, vertical_data
import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import NeuralNetwork, LossCategoricalCrossentropy
from Evaluation import Accuracy
from Optimizer import OptimizerSGD, Trainer

X, y = spiral_data(samples=100, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# plt.scatter(X[:, 0 ], X[:, 1 ], c = y, s = 40 , cmap = 'brg' )
# plt.show()

neural_network = NeuralNetwork(2, 1, 64, 3)

# neural_network.forward(X)
# output = neural_network.output
# predictions = np.argmax(output, axis = 1)
# plt.scatter(X[:, 0 ], X[:, 1 ], c = predictions, s = 40 , cmap = 'brg' )
# plt.show()

optimizer = OptimizerSGD(decay=1e-3, momentum=0.5)
trainer = Trainer(optimizer)
trainer.set_training_data(X, y)
trainer.set_validation_data(X_test, y_test)
trainer.set_neural_network(neural_network)
trainer.set_loss_function(LossCategoricalCrossentropy())
trainer.train(10000)

# neural_network.forward(X)
# output = neural_network.output
# predictions = np.argmax(output, axis = 1)
# plt.scatter(X[:, 0 ], X[:, 1 ], c = predictions, s = 40 , cmap = 'brg' )
# plt.show()