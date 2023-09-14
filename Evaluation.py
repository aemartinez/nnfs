import numpy as np

class Accuracy:

    def calculate(self, output, y):

        predictions = np.argmax(output, axis = 1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis = 1)
        
        return np.mean(predictions == y)