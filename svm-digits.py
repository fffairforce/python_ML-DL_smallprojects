import matplotlib.pyplot as plt
import numpy as np
import random


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        """
    数据结构，维护所有需要操作的值
        :param dataMatIn:数据矩阵
        :param classLabels:数据标签
        :param C:松弛变量
        :param toler:容错率
        :param kTup:包含核函数信息的元组,第一个参数存放核函数类别，第二个参数存放必要的核函数需要用到的参数
        """
        self.X = dataMatIn
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))  # 根据矩阵行数初始化alpha参数为0
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))   # 根据矩阵行数初始化误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        self.K = np.mat(np.zeros((self.m, self.m)))  # 初始化核K
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def kernelTrans(X, A, kTup):
    """
     通过核函数将数据转换更高维的空间
    :param X:
    :param A:
    :param kTup:
    :return:
    """
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRaw = X[j, :] - A
            K[j] = deltaRaw * deltaRaw.T
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('核函数无法识别')
    return K


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def calcEk(oS, k):
    """
    计算误差
    :param oS:数据结构
    :param k:标号为k的数据
    :return: Ek - 标号为k的数据误差
    """
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T*oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJrand(i, m):
    """
    函数说明:随机选择alpha_j的索引值
    :param i:alpha_i的索引值
    :param m:alpha参数个数
    :return:alpha_j 的索引值
    """
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def selectJ(i, oS, Ei):
    """
    内循环启发方式2
    :param i:标号为i的数据的索引值
    :param oS:数据结构
    :param Ei:标号为i的数据误差
    :return:
            j, maxK - 标号为j或maxK的数据的索引值
            Ej - 标号为j的数据误差
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [i, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
                Ek = calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k; maxDeltaE = deltaE, Ej = Ek
    return maxK, Ej


def updateEk(oS, k):
    """
    计算Ek,并更新误差缓存
    Parameters：
        oS - 数据结构
        k - 标号为k的数据的索引值
    Returns:
        无
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aJ = L
    return aj


def innerL(i, oS):
    """
        优化的SMO算法
        Parameters：
            i - 标号为i的数据的索引值
            oS - 数据结构
        Returns:
            1 - 有任意一对alpha值发生变化
            0 - 没有任意一对alpha值发生变化或变化太小
        """
    # 步骤1：计算误差Ei
    Ei = calcEk(oS, i)
    # 优化alpha,设定一定的容错率。
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = selectJ(i, oS, Ei)
        # 保存更新前的aplpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 步骤2：计算上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # 步骤3：计算eta
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新Ej至误差缓存
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("alpha_j变化太小")
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (
                    alphaJold - oS.alphas[j])
        # 更新Ei至误差缓存
        updateEk(oS, i)
        # 步骤7：更新b_1和b_2
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[
            i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[
            i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        # 步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabel, C, toler, maxIter, kTup = ('lin', o)):
    """
        完整的线性SMO算法
        Parameters：
            dataMatIn - 数据矩阵
            classLabels - 数据标签
            C - 松弛变量
            toler - 容错率
            maxIter - 最大迭代次数
            kTup - 包含核函数信息的元组
        Returns:
            oS.b - SMO算法计算的b
            oS.alphas - SMO算法计算的alphas
        """
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabel).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
    if entireSet:
        for i in range(oS.m):
            alphaPairsChanged += innerL(i, oS)
            print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
        iter += 1
    else:  # 遍历非边界值
        nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[
        0]  # 遍历不在边界0和C的alpha
    for i in nonBoundIs:
        alphaPairsChanged += innerL(i, oS)
        print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
    iter += 1
    if entireSet:  # 遍历一次后改为非边界遍历
        entireSet = False
    elif (alphaPairsChanged == 0):  # 如果alpha没有更新,计算全样本遍历
        entireSet = True
    print("迭代次数: %d" % iter)
    return oS.b, oS.alphas  # 返回SMO算法计算的b和alphas


def img2vector(filename):
    """
    将32x32的二进制图像转换为1x1024向量。
    Parameters:
        filename - 文件名
    Returns:
        returnVect - 返回的二进制图像的1x1024向量
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def loadImages(dirName):
    """
    加载图片
    Parameters:
        dirName - 文件夹的名字
    Returns:
        trainingMat - 数据矩阵
        hwLabels - 数据标签
    """
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    """
    测试函数
    Parameters:
        kTup - 包含核函数信息的元组
    Returns:
        无
    """
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10, kTup)
    datMat = np.mat(dataArr);
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd];
    print("支持向量个数:%d" % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("训练集错误率: %.2f%%" % (float(errorCount) / m))
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = np.mat(dataArr);
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("测试集错误率: %.2f%%" % (float(errorCount) / m))


if __name__ == '__main__':
    testDigits()