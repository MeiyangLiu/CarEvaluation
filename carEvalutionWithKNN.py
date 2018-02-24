# -*- coding: utf-8 -*-
from numpy import *
import operator 
import dataPreprocess as dataPreprocess

#基于k-NN的分类器的实现
def classifyCarWithKNN(CarData, DataSet, Labels, k):
    DataSetSize = DataSet.shape[0] #获取矩阵第一纬度的长度
    DiffMat = tile(CarData, (DataSetSize, 1)) - DataSet
    sqDiffMat = DiffMat**2
    sqDistances = sqDiffMat**0.5  #计算欧式距离
    distances = sqDistances.sum(axis=1) #矩阵行相加。生成新矩阵
    sortedDistIndicies = distances.argsort() #返回矩阵中的数组从小到大的下标值，返回新矩阵
    classCount = {}  #初始化新字典
    for i in range(k):
        voterLabel = Labels[sortedDistIndicies[i]]
        classCount[voterLabel] = classCount.get(voterLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True) #排序
    return sortedClassCount[0][0]

#计算混淆矩阵	
def getConfusionMatrix(cMat,real,predict):    
    if(real == 1 and predict == 1):
	    cMat[0][0] += 1
    if(real == 1 and predict == 2):
        cMat[0][1] += 1
    if(real == 1 and predict == 3):
        cMat[0][2] += 1
    if(real == 1 and predict == 4):
        cMat[0][3] += 1
    if(real == 2 and predict == 1):
        cMat[1][0] += 1
    if(real == 2 and predict == 2):
        cMat[1][1] += 1
    if(real == 2 and predict == 3):
        cMat[1][2] += 1
    if(real == 2 and predict == 4):
        cMat[1][3] += 1
    if(real == 3 and predict == 1):
        cMat[2][0] += 1
    if(real == 3 and predict == 2):
        cMat[2][1] += 1
    if(real == 3 and predict == 3):
        cMat[2][2] += 1
    if(real == 3 and predict == 4):
        cMat[2][3] += 1
    if(real == 4 and predict == 1):
        cMat[3][0] += 1
    if(real == 4 and predict == 2):
        cMat[3][1] += 1
    if(real == 4 and predict == 3):
        cMat[3][2] += 1
    if(real == 4 and predict == 4):
        cMat[3][3] += 1
    #print(cMat)
    return cMat

#根据混淆矩阵计算accuracy，precision，recall,F1
def getEvaluation(confusion_matrix,totalNumOfTestData):
    accu = [0,0,0,0]  
    column = [0,0,0,0]  
    line = [0,0,0,0]  
    accuracy = 0  
    recall = 0  
    precision = 0  
    for i in range(0,4):  
        accu[i] = confusion_matrix[i][i] #遍历对角线，获得预测正确的值
    for i in range(0,4):  
       for j in range(0,4):  
           column[i]+=confusion_matrix[j][i] #求出每列的和
    for i in range(0,4):  
       for j in range(0,4):  
           line[i]+=confusion_matrix[i][j]  #求出每行的和
    for i in range(0,4):  
        accuracy += float(accu[i])/totalNumOfTestData #计算accuracy
    for i in range(0,4):  
        if column[i] != 0:  
            precision +=float(accu[i])/column[i]  #每种类别precision之和
    precision = precision / 4  # precision的算数平均
    for i in range(0,4):  
        if line[i] != 0:  
            recall +=float(accu[i])/line[i]   #每种类别recall之和
    recall = recall / 4  # recall的算数平均
    f1_score = (2 * (precision * recall)) / (precision + recall) #计算F1
    print('the totalNumOfTestData is %s' % totalNumOfTestData)
    print('the accu is %s' % accu)	
    print('accuracy  = %s' % accuracy)    
    print('precision = %s' % precision)
    print('recall    = %s' % recall)
    print('f1_score  = %s' % f1_score)	

#分类器训练与测试
def carEvaClassTest():
    basePer = 0.2 #测试基数，选取文本中20%的数据进行测试
    CarDataMat, CarLabels = dataPreprocess.dataCarMatrix() 
    normMat, ranges, minVals = dataPreprocess.autoNorm(CarDataMat) #进行数据归一化
    totalLength = normMat.shape[0]  #读取数据的列长度
    numTestVecs = int(totalLength * basePer) #确定测试的数量
    confusion_matrix=[
	[0,0,0,0],
	[0,0,0,0],
	[0,0,0,0],
	[0,0,0,0]]
    for i in range(numTestVecs):  #进行循环测试
        result = classifyCarWithKNN(normMat[i, :], normMat[numTestVecs:totalLength, :], \
                             CarLabels[numTestVecs:totalLength], 6) #通过分类器进行判断
        confusion_matrix = getConfusionMatrix(confusion_matrix,CarLabels[i],result) 
    print('confusion_matrix  = %s' % confusion_matrix)	
    getEvaluation(confusion_matrix,numTestVecs)

carEvaClassTest()
