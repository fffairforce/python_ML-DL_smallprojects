from math import log
import operator


def createDataSet():
    """
        # 年龄：0代表青年，1代表中年，2代表老年；
        # 有工作：0代表否，1代表是；
        # 有自己的房子：0代表否，1代表是；
        # 信贷情况：0代表一般，1代表好，2代表非常好；
        # 类别(是否给贷款)：no代表否，yes代表是
:return:
        dataSet - 数据集
        labels - 分类属性
    """
    dataSet = [[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def calcShannonEnt(dataSet):
    """
函数说明:计算给定数据集的经验熵(香农熵)
    :param dataSet:
    :return shannonEnt:
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentlabel = featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
            labelCounts[currentlabel] += 1
            shannonEnt = 0.0
    for key in labelCounts:
                prob = float(labelCounts[key])/numEntries
                shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):  # 遍历所有特征
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建set集合{},元素不可重复
        newEntropy = 0.0  # 经验条件熵
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy  # 信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
        if (infoGain > bestInfoGain):  # 计算信息增益
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回信息增益最大的特征的索引值


def majorcnt(classlist):
    """
统计classList中出现此处最多的元素(类标签)
    :param classlist:- 类标签列表
    :return sortedclasscount:- 出现此处最多的元素(类标签)
    """
    classcount = {}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote] = 0
            classcount[vote] += 1
    sortedclasscount = sorted(classcount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedclasscount


def createtree(dataset, labels, featlabels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1 or len(labels) == 0:
        return majorcnt(classlist)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatlabel = labels[bestFeat]
    featlabels.append(bestFeatlabel)
    mytree = {bestFeatlabel: {}}
    del(labels[bestFeat])
    featvalues = [example[bestFeat] for example in dataset]
    uniqusvals = set(featvalues)
    for value in uniqusvals:
        sublabels = labels[:]
        mytree[bestFeatlabel][value] = createtree(splitDataSet(dataset, bestFeat, value), sublabels, featlabels)
    return mytree


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels=[]
    mytree = createtree(dataSet, labels, featLabels)
    #print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))
    print(mytree)