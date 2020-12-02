from SVM import LinearSVM
import numpy as np
from matplotlib import pyplot as plt

dataset = [
    [[0.3858, 0.4687], 1],
    [[0.4871, 0.611], -1],
    [[0.9218, 0.4103], -1],
    [[0.7382, 0.8936], -1],
    [[0.1763, 0.0579], 1],
    [[0.4057, 0.3529], 1],
    [[0.9355, 0.8132], -1],
    [[0.2146, 0.0099], 1]
]
##linearSVM = LinearSVM.LinearSVM(dataset.__len__(), dataset[0][0].__len__())
linearSVM = LinearSVM(dataset.__len__(), dataset[0][0].__len__())#将类实例化
linearSVM.train(dataset, 100)
print(linearSVM)

for record in dataset:
    vector = record[0]
    label = record[-1]
    if label == 1:
        plt.plot(vector[0], vector[1], 'r-o')
    else:
        plt.plot(vector[0], vector[1], 'g-o')

    predict = linearSVM.predict(vector)
    print(record.__str__() + predict.__str__() + '\n')

x1 = np.linspace(0, 1, 50)
x2 = (-linearSVM.bias - linearSVM.weight_vec[0] * x1) / linearSVM.weight_vec[1]
plt.plot(x1, x2)
plt.show()