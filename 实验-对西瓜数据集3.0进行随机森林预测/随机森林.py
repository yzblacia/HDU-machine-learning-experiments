import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
import os
os.environ['PATH'] = os.pathsep + r'G:\Graphviz\bin'

df = pd.read_excel('西瓜数据集3.0.xls')
df.head(10)

df['色泽'] = df['色泽'].map({'浅白': 1, '青绿': 2, '乌黑': 3})
df['根蒂'] = df['根蒂'].map({'稍蜷': 1, '蜷缩': 2, '硬挺': 3})
df['敲声'] = df['敲声'].map({'清脆': 1, '浊响': 2, '沉闷': 3})
df['纹理'] = df['纹理'].map({'清晰': 1, '稍糊': 2, '模糊': 3})
df['脐部'] = df['脐部'].map({'平坦': 1, '稍凹': 2, '凹陷': 3})
df['触感'] = np.where(df['触感'] == "硬滑", 1, 2)
df['好瓜'] = np.where(df['好瓜'] == "是", 1, 0)
x_train = df[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']]
y_train = df['好瓜']

# 构建模型并训练
gini = tree.DecisionTreeClassifier()
gini = gini.fit(x_train, y_train)
# 实现决策树的可视化
gini_data = tree.export_graphviz(gini
                                 , feature_names=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
                                 , class_names=["好瓜", "坏瓜"]
                                 , filled=True
                                 , rounded=True
                                 )
gini_graph = graphviz.Source(gini_data,encoding="utf-8")
print(gini_graph)
gini_graph.render('决策树可视化')

with open('Source.gv',encoding='utf-8') as fj:
    source=fj.read()

dot=graphviz.Source(source)
dot.view()
