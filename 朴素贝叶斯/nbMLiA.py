from NaiveBayes import *
'''
主函数
'''
def main():
    path = "D:\\AI\\data\\" #文件目录
    training_sample = 'trainingSet.txt' #训练数据文件
    testing_sample = 'testingSet.txt' #测试数据文件
    trainingSet,label = loadDataSet(path,training_sample) #获取训练数据
    testingSet,label = loadDataSet(path,testing_sample) #获取测试数据
    h = getResult(trainingSet,testingSet) #计算结果向量
    plotBestFit(testingSet,label,h) #图形化展示


'''
程序入口
'''
if __name__ == '__main__':
    main()