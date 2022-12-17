import numpy as np
from 决策树_预处理 import pre

df = pre()
print('----------建立决策树模型实现预测部分----------')
x_cols = [col for col in df.columns if col != 'Target']
y_col = 'Target'

from sklearn.model_selection import train_test_split
from collections import Counter

X_train, X_test, y_train, y_test = train_test_split(df[x_cols],
                                                    df[y_col],
                                                    test_size=0.1,  # 分割比例
                                                    random_state=42,  # 随机数种子
                                                    shuffle=True,  # 是否打乱顺序
                                                    stratify=df[y_col])  # 指定以Target的比例做分层抽样
print('Distribution of y_train {}'.format(Counter(y_train)))
print('Distribution of y_test {}'.format(Counter(y_test)))

# 引入StandardScaler标准化工具库
from sklearn.preprocessing import StandardScaler

# 对训练集和测试集做标准化
std_scaler = StandardScaler().fit(df[x_cols])
X_train_std = std_scaler.transform(X_train)
X_test_std = std_scaler.transform(X_test)

# 引入决策树和特征选择的库
from sklearn import tree

# 声明决策树模型
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train_std, y_train)
score = clf.score(X_test_std, y_test)

# 特征重要性
print(clf.feature_importances_)

# predict返回每个测试样本的分类/回归结果
y_pre = clf.predict(X_test_std)

print(y_pre)

# 引入评价指标的库
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

print('accuracy of lr_model score', accuracy_score(y_test, y_pre))
print('f1_score of lr_model score', f1_score(y_test, y_pre))
print('recall_score of lr_model score', recall_score(y_test, y_pre))
print('precision_score of lr_model score', precision_score(y_test, y_pre))
print(score)

##### DT调参
from sklearn.model_selection import GridSearchCV

gini_thresholds = np.linspace(0, 0.5, 20)
parameters = {'splitter': ('best', 'random')
    , 'criterion': ("gini", "entropy")
              # ,"max_depth":[*range(1,10)]
              # ,'min_samples_leaf':[*range(1,50,5)]
              # ,'min_impurity_decrease':[*np.linspace(0,0.5,20)]
              }
clf = tree.DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf, parameters, cv=10)
GS.fit(X_train_std, y_train)
print(GS.best_params_)
print(GS.best_score_)
