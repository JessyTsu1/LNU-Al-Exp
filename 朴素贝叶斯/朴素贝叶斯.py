# encoding:utf-8

import pandas as pd
import numpy  as np


class NaiveBayes:
    def __init__(self):
        self.model = {}  # key 为类别名 val 为字典PClass表示该类的该类，PFeature:{}对应对于各个特征的概率

    def calEntropy(self, y):  # 计算熵
        valRate = #此处补全 # 频次汇总 得到各个特征对应的概率
        valEntropy = np.inner(valRate, np.log2(valRate)) * -1
        return valEntropy

    def fit(self, xTrain, yTrain=pd.Series()):
        if not #此处补全:  # 如果不传，自动选择最后一列作为分类标签
            xTrain = pd.concat([xTrain, yTrain], axis=1)
        self.model = self.buildNaiveBayes(xTrain)
        return self.model

    def buildNaiveBayes(self, xTrain):
        yTrain = xTrain.iloc[:, -1]

        yTrainCounts = yTrain.value_counts()  # 频次汇总 得到各个特征对应的概率

        yTrainCounts = #此处补全  # 使用了拉普拉斯平滑
        retModel = {}
        for nameClass, val in yTrainCounts.items():
            retModel[nameClass] = {'PClass': val, 'PFeature': {}}

        propNamesAll = xTrain.columns[:-1]
        allPropByFeature = {}
        for nameFeature in propNamesAll:
            allPropByFeature[nameFeature] = list(xTrain[nameFeature].value_counts().index)
        # print(allPropByFeature)
        for nameClass, group in xTrain.groupby(xTrain.columns[-1]):
            for nameFeature in propNamesAll:
                eachClassPFeature = {}
                propDatas = group[nameFeature]
                propClassSummary = #此处补全  # 频次汇总 得到各个特征对应的概率
                for propName in allPropByFeature[nameFeature]:
                    if not propClassSummary.get(propName):
                        #此处补全 = 0  # 如果有属性灭有，那么自动补0
                Ni = len(allPropByFeature[nameFeature])
                propClassSummary = propClassSummary.apply(lambda x: (x + 1) / (propDatas.size + Ni))  # 使用了拉普拉斯平滑
                for nameFeatureProp, valP in propClassSummary.items():
                    eachClassPFeature[nameFeatureProp] = valP
                retModel[nameClass]['PFeature'][nameFeature] = eachClassPFeature

        return retModel

    def predictBySeries(self, data):
        curMaxRate = None
        curClassSelect = None
        for nameClass, infoModel in self.model.items():
            rate = 0
            rate += np.log(infoModel['PClass'])
            PFeature = infoModel['PFeature']

            for nameFeature, val in data.items():
                propsRate = PFeature.get(nameFeature)
                if not propsRate:
                    continue
                rate += #此处补全  # 使用log加法避免很小的小数连续乘，接近零
                # print(nameFeature, val, propsRate.get(val, 0))
            # print(nameClass, rate)
            if curMaxRate == None or rate > curMaxRate:
                curMaxRate = rate
                curClassSelect = nameClass

        return curClassSelect

    def predict(self, data):
        if isinstance(data, pd.Series):
            return self.predictBySeries(data)
        return data.apply(lambda d: self.predictBySeries(d), axis=1)


dataTrain = pd.read_csv("xiguadata.csv", encoding="UTF-8")

naiveBayes = NaiveBayes()
treeData = naiveBayes.fit(dataTrain)

import json

print(json.dumps(treeData, ensure_ascii=False))

pd = pd.DataFrame({'预测值': naiveBayes.predict(dataTrain), '正取值': dataTrain.iloc[:, -1]})
print(pd)
print('正确率:%f%%' % (pd[pd['预测值'] == pd['正取值']].shape[0] * 100.0 / pd.shape[0]))