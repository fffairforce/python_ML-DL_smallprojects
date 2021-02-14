from math import log
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt


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
        # print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
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


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va='center', ha='center', rotation=30)


def getNumLeafs(myTree):
    numLeafs = 0                                                #初始化叶子
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':                #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0                                                #初始化决策树深度
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':                #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth            #更新层数
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc",
                          size=14)  # 设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            # 绘制结点
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType,
                            arrowprops=arrow_args, FontProperties=font)


def plotTree(mytree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle='sawtooth', fc='0.8')
    leafNode = dict(boxstyle='round4', fc='0.8')
    numLeafs = getNumLeafs(mytree)
    depth = getTreeDepth(mytree)
    firstStr = next(iter(mytree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')                            # 创建fig
    fig.clf()                                                         # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))         # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))              # 获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0           # x偏移
    plotTree(inTree, (0.5, 1.0), '')                                # 绘制决策树
    plt.show()


def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    mytree = createtree(dataSet, labels, featLabels)
    # print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))
    # print(mytree)
    # createPlot(mytree)
    testVec = [0, 1]
    result = classify(mytree, featLabels, testVec)
    if result == 'yes':
        print('loan')
    if result == 'no':
        print('dont loan')
