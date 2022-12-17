import pandas as pd
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

train_data = pd.read_csv('train_local.csv')
test_data = pd.read_csv('test_local.csv')
all_data = pd.concat([train_data, test_data], ignore_index=True)
# 数据初步分析

all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
Title_dict = {}
Title_dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_dict.update(dict.fromkeys(['Don', 'Str', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
print(Title_dict)
all_data['Title'] = all_data['Title'].map(Title_dict)

all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Desk'] = all_data['Cabin'].str.get(0)

Ticket_count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x: Ticket_count[x])


def Ticket_depart(s):
    if (s >= 2) and (s <= 4):
        return 2
    elif ((s > 4) and (s <= 8)) or (s == 1):
        return 1
    elif s > 8:
        return 0


all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_depart)
# 缺失值清洗
# Age缺失值为263，用Sex,Title,Pclass三个特征构建随机森林模型，填充年龄缺失值
from sklearn.ensemble import RandomForestRegressor

age_df = all_data[['Age', 'Pclass', 'Sex', 'Title']]
age_df = pd.get_dummies(age_df)
# 只会将分类变量变成虚拟变量，不会将连续的数值变量变成虚拟变量

known_age = age_df[age_df.Age.notnull()].values
unknown_age = age_df[age_df.Age.isnull()].values

y = known_age[:, 0]
X = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)
predictedAge = rfr.predict(unknown_age[:, 1::])
all_data.loc[(all_data.Age.isnull()), 'Age'] = predictedAge

all_data.groupby(by=['Pclass', 'Embarked']).Fare.median()  # 用fare的中位数
# embarked 也有2个缺失值，根据这个个体的其他信息，直接填充
all_data['Embarked'] = all_data['Embarked'].fillna('C')
# fare 的缺失值，用中位数填充
fare = all_data[(all_data['Embarked'] == 'S') & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare'] = all_data['Fare'].fillna(fare)
# 2.异常值处理
#
# 多人家庭中没有获救的女性和儿童为异常值
all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0].strip())
Surname_count = dict(all_data['Surname'].value_counts())
all_data['Family_count'] = all_data['Surname'].apply(lambda x: Surname_count[x])

# 找出所有处于多人家庭的儿童和妇女和成年男性
Female_Child_Count = all_data.loc[
    (all_data['Family_count'] >= 2) & ((all_data['Age'] <= 12) | (all_data['Sex'] == 'female'))]
Male_Adult_Count = all_data.loc[(all_data['Family_count'] >= 2) & (all_data['Age'] > 12) & (all_data['Sex'] == 'male')]

Female_Child = pd.DataFrame(Female_Child_Count.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns = ['GroupCount']

Female_Child_Count = Female_Child_Count.groupby('Surname')['Survived'].mean()
Dead_List = set(Female_Child_Count[Female_Child_Count.apply(lambda x: x == 0)].index)

Male_Adult_Count = Male_Adult_Count.groupby('Surname')['Survived'].mean()
Survived_List = set(Male_Adult_Count[Male_Adult_Count.apply(lambda x: x == 1)].index)

# 将测试集中所有幸存组的成员改成女性和儿童，将遇难组的都改成男性
train = all_data.loc[all_data['Survived'].notnull()]
test = all_data.loc[all_data['Survived'].isnull()]

test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Age'] = 60
test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Age'] = 5
test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Title'] = 'Miss'

# 特征转换
all_data = pd.concat([train, test])
all_data = all_data[
    ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'Desk', 'TicketGroup']]
all_data = pd.get_dummies(all_data)
train = all_data[all_data['Survived'].notnull()]
test = all_data[all_data['Survived'].isnull()].drop('Survived', axis=1)
X = train.values[:, 1:]
y = train.values[:, 0]
X_test = test
y_test = pd.read_csv('test_local_label.csv').values[:, 1]
print(train.info())
print(test.info())

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

pipe = Pipeline([('select', SelectKBest(k=20)),
                 ('classify', RandomForestClassifier(random_state=10, max_features='sqrt'))
                 ])
parm_test = {'classify__n_estimators': list(range(20, 50, 2)),
             'classify__max_depth': list(range(3, 20, 1)),
             }

gsearch = GridSearchCV(estimator=pipe, param_grid=parm_test, scoring='roc_auc', cv=10)
gsearch.fit(X, y)
print(gsearch.best_params_, gsearch.best_score_)

from sklearn.pipeline import make_pipeline

select = SelectKBest(k=20)
clf = RandomForestClassifier(random_state=10, warm_start=True,
                             n_estimators=26,
                             max_depth=8,
                             max_features='sqrt'
                             )
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)

from sklearn import model_selection, metrics

cv_score = model_selection.cross_val_score(pipeline, X_test, y_test, cv=10)
print('CV score: Mean-%.7g | Std -%.7g' % (np.mean(cv_score), np.std(cv_score)))

# predictions = gsearch.predict(test)
# submission = pd.DataFrame({'PassengerID': test.index + 1, 'Survived': predictions.astype(np.int32)})
# submission.to_csv(r'./submission.csv', index=False)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X, y)
Y_pred = logreg.predict(test)
acc_log = round(logreg.score(X_test, y_test) * 100, 2)
print('acc_log: %f', acc_log)
# Support Vector Machines
svc = SVC()
svc.fit(X, y)
Y_pred = svc.predict(test)
acc_svc = round(svc.score(X_test, y_test) * 100, 2)
print('acc_svc: %f', acc_svc)
#  k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
Y_pred = knn.predict(test)
acc_knn = round(knn.score(X_test, y_test) * 100, 2)
print('acc_knn: %f', acc_knn)
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X, y)
Y_pred = gaussian.predict(test)
acc_gaussian = round(gaussian.score(X_test, y_test) * 100, 2)
print('acc_gaussian : %f', acc_gaussian)
# Perceptron
perceptron = Perceptron()
perceptron.fit(X, y)
Y_pred = perceptron.predict(test)
acc_perceptron = round(perceptron.score(X_test, y_test) * 100, 2)
print('acc_perceptron : %f', acc_perceptron)
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X, y)
Y_pred = linear_svc.predict(test)
acc_linear_svc = round(linear_svc.score(X_test, y_test) * 100, 2)
print('acc_linear_svc : %f', acc_linear_svc)
# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X, y)
Y_pred = sgd.predict(test)
acc_sgd = round(sgd.score(X_test, y_test) * 100, 2)
print('acc_sgd : %f', acc_sgd)
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X, y)
Y_pred = decision_tree.predict(test)
acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
print('acc_decision_tree : %f', acc_decision_tree)
# submission = pd.DataFrame({'PassengerID': test.index + 1, 'Survived': Y_pred.astype(np.int32)})
# submission.to_csv(r'./submission.csv', index=False)
