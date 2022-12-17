import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test.head()

train.head()

train.shape , test.shape

train.info()

test.info()

train2 = train.loc[:,['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare']]
train2.head()

test2 = test.loc[:,['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare']]
test2.head()

train2.info()

test2.info()

#取中位数填充年龄Age
age = train2['Age'].median()
train2.loc[train2.loc[:,'Age'].isnull(),"Age"] = age
test2.loc[test2.loc[:,'Age'].isnull(),"Age"] = age
train2['Age']=train2['Age'].astype(np.int64)
test2['Age']=test2['Age'].astype(np.int64)

#取众数填充票价Fale
Fare = test2['Fare'].mode()
Fare
test2.loc[test2['Fare'].isnull(),'Fare']= Fare[0]

test2.info(),train2.info()

#字符串装换为数值
train2.loc[:,'Sex']
test2.loc[:,'Sex']

train2['Sex']=train2['Sex'].map({'female': 0, 'male': 1}).astype(np.int)
train2.head()

test2['Sex']=test2['Sex'].map({'female': 0, 'male': 1}).astype(np.int)
test2.head()

train2.info(),test2.info()

#familysize
train2['familysize'] = train2['SibSp']+train2['Parch']+1
test2['familysize'] = test2['SibSp']+test2['Parch']+1

#isalone
train2['isalone'] = 0
train2.loc[train2['familysize']==1,'isalone'] = 1

#isalone
test2['isalone'] = 0
test2.loc[test2['familysize']==1,'isalone'] = 1

#去掉sibsp、Parch列
train2 = train2.drop(['SibSp','Parch'],axis=1)
test2 = test2.drop(['SibSp','Parch'],axis=1)

train2.head()

test2.head()

train2.describe()

test2.describe()

train2.head()

#统计频率用交叉表
pclass = pd.crosstab(train2.Pclass,train2.Survived,margins=True)
pclass

pclass.plot.bar()

#统计性别
sex = pd.crosstab(train2.Sex,train2.Survived)
sex

sex.plot.bar()

#isalone
isalone = pd.crosstab(train2.isalone,train2.Survived)
isalone

isalone.plot.bar()

familysize = pd.crosstab(train2.familysize,train2.Survived)
familysize

familysize.plot.bar()

#不同船票价格区间存活率
#利用面元划分
#fare = pd.crosstab(train2.Fare,train2.Survived, margins=True)
#fare.plot.bar() 分类太多不易比较，需要用到面元
bins = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500,600]
fare = pd.cut(train2.Fare,bins)

temp = pd.DataFrame()
temp['Survived'] = train2.Survived
temp['Fare'] = fare
temp.head()
    
# temp.loc[train2.Fare > 500]

fare = pd.crosstab(temp.Fare,temp.Survived)
fare.plot.bar()

train2.describe()

test2.info()

#数据标准化
from sklearn import preprocessing
train2.Age = preprocessing.scale(train2.Age.values)
test2.Age = preprocessing.scale(test2.Age.values)
train2.Fare = preprocessing.scale(train2.Fare.values)
test2.Fare = preprocessing.scale(test2.Fare.values)

train2.describe()

train2.head()

#特征x 
x = train2.drop(['PassengerId','Survived'],axis=1).values
x

#标签y
y = train2.Survived.values
y

#建立训练集和测试集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#建立分类器

#决策树
#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()

#逻辑回归
from sklearn.linear_model import LogisticRegression
my_classifier = LogisticRegression()

#KNN
#from sklearn.neighbors import KNeighborsClassifier
#my_classifier = KNeighborsClassifier()


#用训练集训练分类器（聚类算法函数不同，且输入值只需要输入x）
my_classifier.fit(x_train,y_train)

#测试集调用预测方法，分类数据
predictions = my_classifier.predict(x_test)
predictions

#比较测试数据预测结果和真是结果，算出正确率
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)

test2.head()

# 预测特征 X2
x2 = test2.drop('PassengerId',axis=1).values
x2

prediction2 = my_classifier.predict(x2)
prediction2

t = pd.DataFrame({'PassengerId': test2.PassengerId, 'Survived': prediction2})
t

t.to_csv('t_LogisticRegression.csv',index=False)

zhenshi = pd.read_csv('gender_submission.csv')
zhenshi.head()

zqv = (zhenshi['Survived'].values == t['Survived'].values).sum() / zhenshi.shape[0]
zqv