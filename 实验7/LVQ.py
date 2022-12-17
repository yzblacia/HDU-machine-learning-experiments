import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_seed(SEED=42):
    np.random.seed(SEED)


def distance(sample, p):
    # 这里用差的平方来表示距离
    d = np.power(sample - p, 2).sum(axis=1)
    cls = d.argmin()
    return cls


def LVQ(samples, labels, k, learning_rate, training):
    data_number = len(samples)
    ran1 = np.random.choice(data_number, k, replace=False)
    p = samples[ran1]
    p_labels = labels[ran1]
    step = training
    while training > 0:
        ran2 = np.random.choice(data_number)
        j = samples[ran2]
        j_label = labels[ran2]
        cls = distance(j, p)
        if j_label == p_labels[cls]:
            p[cls] = p[cls] + learning_rate * (j - p[cls])
        else:
            p[cls] = p[cls] - learning_rate * (j - p[cls])
        training -= 1

    clusters = [[] for i in range(k)]
    for sample in samples:
        ci = distance(sample, p)
        clusters[ci].append(sample)
    clusters_show(clusters, step)

    return p


def clusters_show(clusters, step):
    color = ["red", "blue", "pink", "orange"]
    marker = ["*", "^", ".", "+"]
    plt.figure(figsize=(8, 8))
    plt.title("training = {}".format(step))
    plt.xlabel("Density", loc='center')
    plt.ylabel("Sugar Content", loc='center')
    # 用颜色区分k个簇的数据样本
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], c=color[i], marker=marker[i], s=150)
    plt.show()


if __name__ == '__main__':
    make_seed()
    # 导入数据
    data = pd.read_csv(r"西瓜数据集4 - LVQ.csv", encoding='gbk')
    samples = data[["密度", "含糖量"]].values
    labels = data[["是否好瓜"]].values
    print("原型向量：")
    print(LVQ(samples, labels, 4, 0.1, 100))
