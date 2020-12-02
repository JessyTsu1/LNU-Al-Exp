# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 13:52:47 2018

@author: Administrator
"""

import numpy as np
import xlrd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def open_excel(file):
    """
    打开excel文件获取数据
    :param file: 文件所在的位置
    :return: 文件数据
    """
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception as e:
        print(str(e))


def split_feature(row):
    """
    将该行特征处理后放入列表中
    :param row:一行特征数据
    :return: 返回数据列表
    """
    app = []
    for i in range(16):
        app = app + [row[i]]
    return app


def loadDataSet(path, training_sample, colnameindex=0, by_name=u'sheet1'):
    """
    加载数据
    :param path: 数据文件存放路径
    :param training_sample: 数据文件名
    :param colnameindex: 文件列名下标
    :param by_name: 表名
    :return: 数据集和类别标签
    """
    dataMat = []  # 定义数据列表
    labelMat = []  # 定义标签列表
    filename = path + training_sample  # 形成特征数据的完整路径
    data = open_excel(filename)  # 打开文件获取数据
    table = data.sheet_by_name(by_name)  # 获得数据表
    nrows = table.nrows  # 得到表数据总行数
    colnames = table.row_values(colnameindex)  # 某一行数据 ['user_id', 'age_range', 'gender', 'merchant_id','label']
    for rownum in range(1, nrows):  # 也就是从Excel第二行开始，第一行表头不算
        row = table.row_values(rownum)  # 取一行数据
        '''
        判断2,3,6列数据是否为空，若为空则丢弃该行数据
        '''
        if row[1] == '' or row[2] == '' or row[5] == '':
            continue
        if row:
            app = split_feature(row)  # 将特征值转化为列表
            dataMat.append(app)
            labelMat.append(float(row[16]))  # 获取类别标签
    return dataMat, labelMat


def show_accuracy(a, b, tip):
    """
    计算准确率
    :param a: 真实类别
    :param b: 预测标签
    :param tip: 描述
    :return: 准确率
    """
    acc = a.ravel() == b.ravel()
    print("%s Accuracy:%.3f" % (tip, np.mean(acc)))


def main():
    """
    主函数
    :return: null
    """
    path = "E:\\"
    training_sample = 'featuredata.xls'  # 特征数据文件
    trainingSet, trainingLabels = loadDataSet(path, training_sample)  # 取特征数据和标签数据
    x = np.array(trainingSet)  # 将数据部分列表（list）格式转化为数组(array)格式
    y = np.array(trainingLabels)  # 将标签部分的列表（list）格式转化为数组格式（array）
    '''
    将数据分为训练数据和测试数据两部分
    x_train 训练数据
    x_test  测试数据
    y_train 训练数据标签
    y_test 测试数据标签
    '''

    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, test_size=0.3)
    # 定义多层感知机分类算法
    clf = MLPClassifier(activation='relu', solver='adam', alpha=0.0001)
    clf.fit(train_data, train_label)  # 利用训练数据训练模型
    hat_test_label = clf.predict(test_data)
    print(classification_report(test_label, hat_test_label))


if __name__ == '__main__':
    """
    程序入口
    """
    main()
