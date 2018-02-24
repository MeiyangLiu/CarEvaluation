# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# 定义数据的路径
ORIGIN_DATA_FILE = './car.data'
RANDOM_DATA_OUT_PATH = './car_random.data'

# 读取文件
def loadData():
    data = pd.read_csv(ORIGIN_DATA_FILE, sep=',', header=None)
    return data

# 删除空格等缺失数据  
def removeMissing(data):
    data.dropna() # 删除NaN
    return data

# 自然语言映射为数字
def nature2num(df):
    level_mapping = {'low': 1,
                     'med': 2,
                     'high': 3,
                     'vhigh': 4}
    size_mapping = {'small': 1,
                    'med': 2,
                    'big': 3}
    class_mapping = {'unacc': 1,
                     'acc': 2,
                     'good': 3,
                     'vgood': 4}
    num_mapping = {'1': 1,
                   '2': 2,
                   '3': 3,
                   '4': 4,
                   '5more': 5,
                   'more': 6}

    df[0] = df[0].map(level_mapping)
    df[1] = df[1].map(level_mapping)
    df[2] = df[2].map(num_mapping)
    df[3] = df[3].map(num_mapping)
    df[4] = df[4].map(size_mapping)
    df[5] = df[5].map(level_mapping)
    df[6] = df[6].map(class_mapping)
    return df

#数据归一化
def autoNorm(DataSet):
    minVals = DataSet.min(0 )#将每列中的最小值放在变量minVals中
    maxVals = DataSet.max(0) #获取数据集中每一列的最大数值
    ranges = maxVals - minVals #最大值与最小的差值
    normDataSet = np.zeros(np.shape(DataSet)) #生成一个与dataSet相同的零矩阵，用于存放归一化后的数据
    m = DataSet.shape[0] #求出dataSet列长度
    normDataSet = DataSet - np.tile(minVals, (m, 1)) #求出oldValue - min
    #把最大最小差值扩充为dataSet同shape，然后作商，是指对应元素进行除法运算，求出归一化数值
    normDataSet = normDataSet / np.tile(ranges, (m,1)) 
    return normDataSet, ranges, minVals

  
# 输出数据，用于测试
def output(data, outpath):
    data.to_csv(outpath, index = False)
    return       

#数据预处理整个过程
def dataCarMatrix():
    data = loadData() #加载car.data数据
    nomis = removeMissing(data) #删除空格等确实数据
    dataNum = nature2num(nomis) #将自然语言映射为数字
    dataRandom = dataNum.sample(frac=1)  #打乱数据顺序，frac=1是选取全部数据   
    output(dataRandom,RANDOM_DATA_OUT_PATH) #测试显示处理后的数据
    allMatrix = dataRandom.as_matrix() #dataframe转化为matrix
    carMatrix = allMatrix[:,0:6] #特征提取，前6列是属性值（特征值）
    lableVector = allMatrix[:,-1] #分类标签label提取，最后一列是label值
    #print(carMatrix) #test
    #print(lableVector) #test
    return carMatrix, lableVector
#for Test
data = loadData() #加载car.data数据
nomis = removeMissing(data) #删除空格等确实数据
dataNum = nature2num(nomis) #将自然语言映射为数字
dataRandom = dataNum.sample(frac=1)  #打乱数据顺序，frac=1是选取全部数据   
output(dataRandom,RANDOM_DATA_OUT_PATH)

	