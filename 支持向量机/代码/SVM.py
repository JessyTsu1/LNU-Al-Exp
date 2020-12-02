import numpy as np


class LinearSVM(object):
    """
    线性SVM的实现类
    """

    def __init__(self, dataset_size=8, vector_size=2):
        """
        初始化函数
        :param dataset_size:数据集规模，用于初始化 ‘拉格朗日乘子’ 的数量
        :param vector_size:特征向量的维度，用于初始化 ‘权重向量’ 的维度
        """
        self.__multipliers = np.zeros(dataset_size, np.float_)
        self.weight_vec = np.zeros(vector_size, np.float_)
        self.bias = 0

    def train(self, dataset, iteration_num):
        """
        训练函数
        :param dataset:数据集，每条数据的形式为（X，y），其中X是特征向量，y是类标号
        :param iteration_num:
        """

        dataset = np.array(dataset)
        for k in range(iteration_num):
            self.__update(dataset, k)

    def __update(self, dataset, k):
        """
        更新函数
        :param dataset:
        :param k:
        """
        for i in range(dataset.__len__() // 2):
            j = (dataset.__len__() // 2 + i + k) % dataset.__len__()
            record_i = dataset[i]
            record_j = dataset[j]
            self.__sequential_minimal_optimization(dataset, record_i, record_j, i, j)
            self.__update_weight_vec(dataset)
            self.__update_bias(dataset)

    def __sequential_minimal_optimization(self, dataset, record_i, record_j, i, j):
        """
        SMO函数，每次选取两条记录，更新对应的‘拉格朗日乘子’
        :param dataset:
        :param record_i:记录i
        :param record_j:记录j
        :param i:
        :param j:
        """
        label_i = record_i[-1]
        vector_i = np.array(record_i[0])
        label_j = record_j[-1]
        vector_j = np.array(record_j[0])

        # 计算出截断前的记录i的‘拉格朗日乘子’unclipped_i
        error_i = np.dot(self.weight_vec, vector_i) + self.bias - label_i
        error_j = np.dot(self.weight_vec, vector_j) + self.bias - label_j
        eta = np.dot(vector_i - vector_j, vector_i - vector_j)
        unclipped_i = self.__multipliers[i] + label_i * (error_j - error_i) / eta

        # 截断记录i的`拉格朗日乘子`并计算记录j的`拉格朗日乘子`
        constant = -self.__calculate_constant(dataset, i, j)
        multiplier = self.__quadratic_programming(unclipped_i, label_i, label_j, i, j)
        if multiplier >= 0:
            self.__multipliers[i] = multiplier
            self.__multipliers[j] = (constant - multiplier * label_i) * label_j

    def __update_bias(self, dataset):
        """
        计算偏置项bias，使用平均值作为最终结果
        :param dataset:
        """
        sum_bias = 0
        count = 0
        for k in range(self.__multipliers.__len__()):
            if self.__multipliers[k] != 0:
                label = dataset[k][-1]
                vector = np.array(dataset[k][0])
                sum_bias += 1 / label - np.dot(self.weight_vec, vector)
                count += 1
        if count == 0:
            self.bias = 0
        else:
            self.bias = sum_bias / count

    def __update_weight_vec(self, dataset):
        """
        计算权重向量
        :param dataset:
        """
        weight_vector = np.zeros(dataset[0][0].__len__())
        for k in range(dataset.__len__()):
            label = dataset[k][-1]
            vector = np.array(dataset[k][0])
            weight_vector += self.__multipliers[k] * label * vector
        self.weight_vec = weight_vector

    def __calculate_constant(self, dataset, i, j):
        label_i = dataset[i][-1]
        label_j = dataset[j][-1]
        dataset[i][-1] = 0
        dataset[j][-1] = 0
        sum_constant = 0
        for k in range(dataset.__len__()):
            label = dataset[k][-1]
            sum_constant += self.__multipliers[k] * label
        dataset[i][-1] = label_i
        dataset[j][-1] = label_j
        return sum_constant

    def __quadratic_programming(self, unclipped_i, label_i, label_j, i, j):
        """
        二次规划，截断`拉格朗日乘子`
        :param unclipped_i:
        :param label_i:
        :param label_j:
        :return:
        """
        multiplier = -1
        if label_i * label_j == 1:
            boundary = self.__multipliers[i] + self.__multipliers[j]
            if boundary >= 0:
                if unclipped_i <= 0:
                    multiplier = 0
                elif unclipped_i < boundary:
                    multiplier = unclipped_i
                else:
                    multiplier = boundary
        else:
            boundary = max(0, self.__multipliers[i] - self.__multipliers[j])
            if unclipped_i <= boundary:
                multiplier = boundary
            else:
                multiplier = unclipped_i
        return multiplier

    def predict(self, vector):
        result = np.dot(self.weight_vec, np.array(vector)) + self.bias
        if result >= 0:
            return 1
        else:
            return -1

    def __str__(self):
        return "multipliers:" + self.__multipliers.__str__() + '\n' + \
               "weight_vector:" + self.weight_vec.__str__() + '\n' + \
               "bias:" + self.bias.__str__()
