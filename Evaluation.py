import numpy as np

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

class Accuracy:

    def calculate(self, output, y):

        predictions = np.argmax(output, axis = 1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis = 1)
        
        return np.mean(predictions == y)