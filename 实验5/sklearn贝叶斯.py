import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 导入数据集
df1 = pd.read_excel('西瓜数据集3.0.xls')
np1 = np.array(df1)
for i in range(np1.shape[1]):
    if i != 7 and i != 8:
        enc = LabelEncoder()
        np1[:, i] = enc.fit_transform(np1[:, i])
        np1[:, i] = np1[:, i].astype('int')
X = np1[:, 1:9]
y = np1[:, 9]
y = y.astype('int')

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)
gnb = GaussianNB().fit(X, y)
acc_score = gnb.score(X, y)
print(acc_score)
y_pred = gnb.predict(X)
yprob = gnb.predict_proba(X)

