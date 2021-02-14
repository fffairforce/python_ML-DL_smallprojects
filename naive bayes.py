import numpy as np
from functools import reduce


def loaddataset():
    postinglist = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                # 切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postinglist, classVec


def setofWords2Vec(vocabList, imputSet):
    returnVec = [0] * len(vocabList)
    for word in imputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('word: %s is not in my vocabulary!' % word)
    return returnVec


def createVocabList(dataSet):
    vocabSet = set([])                   # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def trainNB0(trainMatrix, trainCategory):
    """

    :param trainMatrix: 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    :param trainCategory: 训练类别标签向量，即loadDataSet返回的classVec
    :return: 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
    """
    numTrainDocs = len(trainMatrix)      #计算训练的文档数目
    numWords = len(trainMatrix[0])      # 计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # initialize matrix
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # 分母初始化为0
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)  # 取对数，防止下溢出
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """

    :param vec2Classify:待分类的词条数组
    :param p0Vec:非侮辱类的条件概率数组
    :param p1Vec:侮辱类的条件概率数组
    :param pClass1:文档属于侮辱类的概率
    :return:
    0 - 属于非侮辱类
    1 - 属于侮辱类
    """
    # p1 = reduce(lambda x, y: x * y, vec2Classify * p1Vec) * pClass1  # multiple multiplication use reduce func tool
    # p0 = reduce(lambda x, y: x * y, vec2Classify * p0Vec) * (1 - pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    print('p0: ', p0)
    print('p1: ', p1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    pass


if __name__ == '__main__':
    postinglist, classVec = loaddataset()
    # print('postingList:\n',postinglist)
    myVocabList = createVocabList(postinglist)
    print('myVocabList:\n', myVocabList)
    trainMat = []
    for postinDoc in postinglist:
        trainMat.append(setofWords2Vec(myVocabList, postinDoc))
    # print('trainMat:\n', trainMat)
    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    # p1V存放的就是各个单词属于侮辱类的条件概率。pAb就是先验概率
