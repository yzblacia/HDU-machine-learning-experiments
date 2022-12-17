from math import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def calculate1(i, label):
    fea_num = 0
    label_num = 0
    for k in range(xigua.shape[0]):
        if xigua[k][-1] == label:
            label_num += 1
            if xigua[k][i] == data_predict[i - 1]:
                fea_num += 1

    p = round(fea_num / label_num, 3)
    return p


def laplacian_calculate1(i, label):
    fea_num = 0
    label_num = 0
    num = 0
    for k in range(xigua.shape[0]):
        if xigua[k][i] > num:
            num = xigua[k][i]
        if xigua[k][-1] == label:
            label_num += 1
            if xigua[k][i] == data_predict[i - 1]:
                fea_num += 1
    p = round((fea_num + 1) / (label_num + num + 1), 3)
    return p


def calculate2(i, label):
    x = data_predict[i - 1]
    num = 0  # 个数
    miu = 0  # 中位数
    var = 0  # 方差
    for k in range(xigua.shape[0]):
        if xigua[k][-1] == label:
            miu += xigua[k][i]
            num += 1
    miu /= num
    for k in range(xigua.shape[0]):
        if xigua[k][-1] == label:
            var += pow(xigua[k][i] - miu, 2)
    var /= num - 1
    return 1 / ((sqrt(2 * pi)) * sqrt(var)) * exp(-(x - miu) * (x - miu) / (2 * var))


def predict(list_a, list_b):
    sum_a = 0.0
    sum_b = 0.0
    for i in range(len(list_a)):
        sum_a = sum_a * list_a[i]
    for j in range(len(list_b)):
        sum_b = sum_b * list_b[j]
    if sum_a >= sum_b:
        result = "好瓜"
    else:
        result = "坏瓜"
    print(result)


if __name__ == "__main__":
    # 使用Pandas导入数据
    filename = "西瓜数据集3.0.xls"
    names = ["编号", "色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖量", "好瓜"]
    data = pd.read_excel(filename, names=names)
    # print(data)
    data_predict = [2, 2, 1, 1, 0, 0, 0.697, 0.460]
    for i in data.columns[1:7]:
        le = LabelEncoder()
        le.fit(data[i])
        data[i] = le.transform(data[i])
    # print(data)
    xigua = np.array(data)
    # print(xigua)

    P_good1 = []
    P_bad1 = []
    P_good1.append(round(8 / 17, 3))
    P_bad1.append(round(9 / 17, 3))
    for i in range(1, 7):
        P_good1.append(calculate1(i, '是'))
        P_bad1.append(calculate1(i, '否'))
    for i in range(7, 9):
        P_good1.append(calculate2(i, '是'))
        P_bad1.append(calculate2(i, '否'))
    print(P_good1)
    print(P_bad1)
    predict(P_good1, P_bad1)

    #添加拉普拉斯修正
    P_good2 = []
    P_bad2 = []
    P_good2.append(round(9 / 19, 3))
    P_bad2.append(round(10 / 19, 3))
    for i in range(1, 7):
        P_good2.append(laplacian_calculate1(i, '是'))
        P_bad2.append(laplacian_calculate1(i, '否'))
    for i in range(7, 9):
        P_good2.append(calculate2(i, '是'))
        P_bad2.append(calculate2(i, '否'))
    print(P_good2)
    print(P_bad2)
    predict(P_good2, P_bad2)