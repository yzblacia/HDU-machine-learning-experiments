import math
import sys

from pandas import *


def seze(_a):
    if _a == "青绿":
        return 0
    if _a == "乌黑":
        return 1
    if _a == "浅白":
        return 2


def gendi(_a):
    if _a == "蜷缩":
        return 0
    if _a == "稍蜷":
        return 1
    if _a == "硬挺":
        return 2


def qiaoshen(_a):
    if _a == "浊响":
        return 0
    if _a == "沉闷":
        return 1
    if _a == "清脆":
        return 2


def wenli(_a):
    if _a == "清晰":
        return 0
    if _a == "稍糊":
        return 1
    if _a == "模糊":
        return 2


def qibu(_a):
    if _a == "凹陷":
        return 0
    if _a == "稍凹":
        return 1
    if _a == "平坦":
        return 2


def chugan(_a):
    if _a == "硬滑":
        return 1
    if _a == "软粘":
        return 0


def prob_cal(list_a, feather, _good):  # 计算在a列表中，西瓜为好或者坏的特征的概率数
    fea_num = 0
    good_num = 0
    for _i in range(len(list_a)):
        if list_a[_i][0] == feather and list_a[_i][1] == _good:
            fea_num = fea_num + 1
        if list_a[_i][1] == _good:
            good_num = good_num + 1
    prob = float(fea_num / good_num)
    return prob


def miu(list_b, _good):  # 计算均值
    feather_con_sum = 0.0
    good_con = 0
    for _i in range(len(list_b)):
        if list_b[_i][1] == _good:
            good_con = good_con + 1
            feather_con_sum = feather_con_sum + list_b[_i][0]
    miu = float(feather_con_sum / good_con)
    return miu


def theta_con(list_b, miu_b, good):  # 计算theta平方
    good_con = 0
    sum_2 = 0.0
    for _i in range(len(list_b)):
        if list_b[_i][1] == good:
            good_con = good_con + 1
            sum_2 = sum_2 + (list_b[_i][0] - miu_b) ** 2

    return float(sum_2 / (good_con - 1))


def prob_cal_con(miu, theta, _x):  # 计算连续值的概率
    return 1 / ((math.sqrt(2 * math.pi)) * math.sqrt(theta)) * math.exp(-(_x - miu) * (_x - miu) / (2 * theta))


def predict(list_a, list_b):
    sum_a = 0.0
    sum_b = 0.0
    for _i in range(len(list_a)):
        sum_a = sum_a * list_a[_i]
    for _j in range(len(list_b)):
        sum_b = sum_b * list_b[_j]
    if sum_a >= sum_b:
        result = "好瓜"
    else:
        result = "坏瓜"
    print(result)


if __name__ == "__main__":
    # 使用Pandas导入csv数据
    filename = "西瓜数据集3.0.xls"
    names = ["编号", "色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖量", "好瓜"]
    data = read_excel(filename, names=names)
    print(data)
    xigua = data.values.tolist()
    print(xigua)
    x1, x2, x3, x4, x5, x6, x7, x8 = [], [], [], [], [], [], [], []
    y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 好瓜0 坏瓜1
    for i in range(17):  # 将西瓜数据集中的数据转化成0，1格式
        x1.append([seze(xigua[i][1]), y[i]])
        x2.append([gendi(xigua[i][2]), y[i]])
        x3.append([qiaoshen(xigua[i][3]), y[i]])
        x4.append([wenli(xigua[i][4]), y[i]])
        x5.append([qibu(xigua[i][5]), y[i]])
        x6.append([chugan(xigua[i][6]), y[i]])
        x7.append([float(xigua[i][7]), y[i]])
        x8.append([float(xigua[i][8]), y[i]])
    datasets_xigua = [x1, x2, x3, x4, x5, x6, x7, x8]
    data_predict = [0, 0, 0, 0, 0, 0, 0.697, 0.460]
    P_good = []
    P_bad = []
    # print(prob_cal_con(0.574,0.129**2,0.697))
    for _a in range(6):
        P_good.append(prob_cal(datasets_xigua[_a], 0, 0))
        P_bad.append(prob_cal(datasets_xigua[_a], 0, 1))

    for _b in range(6, 8):
        miu_good = miu(datasets_xigua[_b], 0)
        miu_bad = miu(datasets_xigua[_b], 1)
        theta_good = theta_con(datasets_xigua[_b], miu_good, 0)
        theta_bad = theta_con(datasets_xigua[_b], miu_bad, 1)
        P_good.append(prob_cal_con(miu_good, theta_good, data_predict[_b]))
        P_bad.append(prob_cal_con(miu_bad, theta_bad, data_predict[_b]))
    print(P_good, P_bad)
    predict(P_good, P_bad)
sys.exit()
