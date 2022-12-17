import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 功能: 设置随机种子, 确保结果可复现
def make_seed(SEED=42):
    np.random.seed(SEED)


# 功能: 计算样本与聚类中心的距离, 返回离簇中心最近的类别
# params: sample: 单个数据样本, centers: k个簇中心
# return: 返回的是当前的样本数据属于那一个簇中心的id或者索引
def distance(sample, centers):
    # 这里用差的平方来表示距离
    d = np.power(sample - centers, 2).sum(axis=1)
    cls = d.argmin()
    return cls


# 功能: 对当前的分类子集进行可视化展示
def clusters_show(clusters, step):
    color = ["red", "blue", "pink", "orange"]
    marker = ["*", "^", ".", "+"]
    plt.figure(figsize=(8, 8))
    plt.title("step: {}".format(step))
    plt.xlabel("Density", loc='center')
    plt.ylabel("Sugar Content", loc='center')
    # 用颜色区分k个簇的数据样本
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], c=color[i], marker=marker[i], s=150)
    plt.show()


# 功能: 根据输入的样本集与划分的簇数，分别返回k个簇样本
# params： data：样本集, k：聚类簇数
# return：返回是每个簇的簇类中心
def k_means(samples, k):
    data_number = len(samples)
    centers_flag = np.zeros((k,))
    # 随机在数据中选择k个聚类中心
    centers = samples[np.random.choice(data_number, k, replace=False)]

    step = 0
    while True:
        # 计算每个样本距离簇中心的距离, 然后分到距离最短的簇中心中
        clusters = [[] for i in range(k)]
        for sample in samples:
            ci = distance(sample, centers)
            clusters[ci].append(sample)

        # 可视化当前的聚类结构
        clusters_show(clusters, step)

        # 分完簇之后更新每个簇的中心点, 得到了簇中心继续进行下一步的聚类
        for i, sub_clusters in enumerate(clusters):
            new_center = np.array(sub_clusters).mean(axis=0)
            # 如果数值有变化则更新, 如果没有变化则设置标志位为1，当所有的标志位为1则退出循环
            # print(centers[i] != new_center)
            if (centers[i] != new_center).any():
                centers[i] = new_center
            else:
                centers_flag[i] = 1
        # print(centers_flag)
        step += 1
        print("step:{}".format(step), "\n", "centers:{}".format(centers))
        if centers_flag.all():
            break

    return centers


if __name__ == '__main__':
    # 导入数据
    data = pd.read_csv(r"西瓜数据集4.csv", encoding='gbk')
    samples = data[["密度", "含糖量"]].values
    for i in range(3, 5):
        make_seed()
        print("分为" + str(i) + "簇的情况：")
        centers = k_means(samples=samples, k=i)
