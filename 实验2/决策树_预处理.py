import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
def pre():
    ##读取数据
    path1 = r"dataset-credit-default.csv"
    df = pd.read_csv(path1, encoding='utf-8')
    print(df.shape)
    # 计算特征相关性
    corr_matrix = df.corr(method='spearman')
    # 可视化
    plt.figure(figsize=(25, 15))
    sns.heatmap(corr_matrix, annot=True)
    plt.show()
    plt.clf()

    # 存储相关性过高的特征对,对于相关性过高的的特征，删除其中一个（根据工程经验，以0.8为界）：
    # 选择出符合内容的单元格对应的行、列标签
    cols_pair_to_drop = []
    for index_ in corr_matrix.index:
        for col_ in corr_matrix.columns:
            if corr_matrix.loc[index_, col_] >= 0.8 and index_ != col_ and (col_, index_) not in cols_pair_to_drop:
                cols_pair_to_drop.append((index_, col_))
    print(cols_pair_to_drop)

    # 丢弃特征对中的一个(删除列元素)
    cols_to_drop = np.unique(
        [col[1] for col in cols_pair_to_drop])  # 对于一维数组或者列表，unique 函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
    df.drop(cols_to_drop, axis=1, inplace=True)
    print(df.shape)  # 34列元素经过删除后变成26列

    # 打印出缺失率最高的前15个特征以及对应的缺失率
    df_missing_stat = pd.DataFrame(df.isnull().sum() / df.shape[0], columns=['missing_rate']).reset_index()
    print(df_missing_stat.sort_values(by='missing_rate', ascending=False)[:15])

    # 从上面箱图可以看出Couple_Year_Income和Couple_L12_Month_Pay_Amount异常值比例都不低，且缺失率很高，可以在删除带有异常值的样本的基础上，一般有两种选择：
    # 尝试用中位数（对数值型变量常用且在该业务场景下，使用中位数也符合实际的分布）来填充缺失值；或者直接删除。本实验尝试第一种。
    # 针对Couple_Year_Income和Couple_L12_Month_Pay_Amount，需要先可视化来判定下如何填充。
    df.boxplot(column=['Couple_Year_Income'])
    plt.show()
    plt.clf()

    # 删除Couple_Year_Income的异常值并用中位数填补缺失值
    item = 'Couple_Year_Income'
    iqr = df[item].quantile(0.75) - df[item].quantile(0.25)
    q_abnormal_L = df[item] < df[item].quantile(0.25) - 1.5 * iqr
    q_abnormal_U = df[item] > df[item].quantile(0.75) + 1.5 * iqr
    # 取异常点的索引
    print(item + '中有' + str(q_abnormal_L.sum() + q_abnormal_U.sum()) + '个异常值')
    item_outlier_index = df[q_abnormal_L | q_abnormal_U].index

    ###删除异常值
    df.drop(index=item_outlier_index, inplace=True)
    print(df.shape)

    # 用中位数填补缺失值
    df[item] = df[item].fillna(df[item].median())

    # 删除Couple_L12_Month_Pay_Amount的异常值并用中位数填补缺失值
    item = 'Couple_L12_Month_Pay_Amount'
    iqr = df[item].quantile(0.75) - df[item].quantile(0.25)
    q_abnormal_L = df[item] < df[item].quantile(0.25) - 1.5 * iqr
    q_abnormal_U = df[item] > df[item].quantile(0.75) + 1.5 * iqr
    print(item + '中有' + str(q_abnormal_L.sum() + q_abnormal_U.sum()) + '个异常值')
    item_outlier_index = df[q_abnormal_L | q_abnormal_U].index
    df.drop(index=item_outlier_index, inplace=True)
    print(df.shape)
    df[item] = df[item].fillna(df[item].median())

    # 名义型变量缺失值，Unit_Kind、Title、Industry和Occupation，先查看其分布。
    # 工作单位性质
    print(df['Unit_Kind'].value_counts())
    # 职务
    print(df['Title'].value_counts())
    # 行业，取值较多，使用柱状图展示，用红色表示
    plt.hist(df['Industry'].value_counts(), color='r')
    # 职务，取值较多，使用柱状图展示，用蓝色表示
    plt.hist(df['Occupation'].value_counts(), color='b')
    plt.show()
    plt.clf()

    ### 查看df中仍有少量缺失值的特征
    null_col = []
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            null_col.append(col)
    print(null_col)

    ###使用众数填充缺失值
    for col_to_fill in null_col:
        df[col_to_fill] = df[col_to_fill].fillna(df[col_to_fill].mode()[0])  # 选择每个特征出现频率对高的第一个属性值进行填充

    ### 删除无分类意义的特征列Cust_No
    del df['Cust_No']

    ### 查看数据集剩余的名称性特征
    con_col = []
    for col in df.columns:
        if df.dtypes[col] == np.object:
            con_col.append(col)

    df['Unit_Kind'] = pd.factorize(df['Unit_Kind'])[0]
    df['Occupation'] = pd.factorize(df['Occupation'])[0]
    print('\n')
    df.info()
    return df
