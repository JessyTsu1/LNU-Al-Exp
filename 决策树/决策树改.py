import numpy as np
import math
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no'], ]
    label = ['no surfacing', 'flippers']
    return dataSet, label


def calcShannonEnt(dataSet):
    # 计算数据信息熵
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in labelCounts:
        shannonEnt = shannonEnt - (labelCounts[key] / numEntries) * math.log2(labelCounts[key] / numEntries)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    # 拆分数据集
    retDataset = []
    for featVec in dataSet:
        if featVec[axis] == value:
            newVec = featVec[:axis]
            newVec.extend(featVec[axis + 1:])
            retDataset.append(newVec)
    return retDataset


def chooseBestFeatureToSplit(dataSet):
    # 计算信息增益
    numFeatures = len(dataSet[0]) - 1
    bestInfoGain = 0
    bestFeature = -1
    baseEntropy = calcShannonEnt(dataSet)
    for i in range(numFeatures):
        allValue = [example[i] for example in dataSet]
        allValue = set(allValue)  # 获取列表中的唯一元素
        newEntropy = 0
        for value in allValue:
            splitset = splitDataSet(dataSet, i, value)
            newEntropy = newEntropy + len(splitset) / len(dataSet) * calcShannonEnt(splitset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    # 创建决策树
    classCount = {}
    for value in classList:
        if value not in classCount: classCount[value] = 0
        classCount[value] += 1
    classCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return classCount[0][0]


def createTree(dataSet, labels):
    # 生成决策树
    classList = [example[-1] for example in dataSet]
    labelsCopy = labels[:]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestLabel = labelsCopy[bestFeature]
    myTree = {bestLabel: {}}
    featureValues = [example[bestFeature] for example in dataSet]
    featureValues = set(featureValues)
    del (labelsCopy[bestFeature])
    for value in featureValues:
        subLabels = labelsCopy[:]
        myTree[bestLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    # 应用建立的决策树进行分类
    currentFeat = list(inputTree.keys())[0]
    secondTree = inputTree[currentFeat]
    try:
        featureIndex = featLabels.index(currentFeat)
    except ValueError as err:
        print('yes')
    try:
        for value in secondTree.keys():
            if value == testVec[featureIndex]:
                if type(secondTree[value]).__name__ == 'dict':
                    classLabel = classify(secondTree[value], featLabels, testVec)
                else:
                    classLabel = secondTree[value]
        return classLabel
    except AttributeError:
        print(secondTree)


if __name__ == "__main__":
    dataset, label = createDataSet()
    myTree = createTree(dataset, label)
    a = [1, 1]
    print(classify(myTree, label, a))
