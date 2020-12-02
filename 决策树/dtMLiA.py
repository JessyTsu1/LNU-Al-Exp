from DecisionTree import *
from valuation import *

def main():
    path = "D:\\AI\\data\\"
    training_sample = 'trainingSet.txt' #训练数据文件
    testing_sample = 'testingSet.txt' #测试数据文件
    trainingSet, trainingLabels = loadDataSet(path,training_sample) #取训练数据
    testingSet, testingLabels = loadDataSet(path,testing_sample) #取测试数据
    feat = featuerSplit(trainingSet)
    h = getResult(testingSet,feat) #预测结果
    treeSet, featurelabels = createDataDic(feat) #构造数据字典
    myTree = createTree(testingSet,treeSet, featurelabels) #创建数据树
    plotBestFit(testingSet,myTree,testingLabels,h) #绘制点和树


'''
程序入口
'''
if __name__=='__main__':
    main()
