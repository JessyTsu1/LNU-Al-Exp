import numpy as np

def MSE(y_true,y_pred,
        sample_weight=None,
        multioutput='uniform_average'):
    sum = 0
    length = len(y_true)
    for i in range(length):
        sum += (y_true[i]-y_pred[i])**2
    sum = sum / length
    return sum





