# 导入相关包
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy.core.umath_tests import inner1d
import pandas as pd
import seaborn as sns

# 设置sns样式
sns.set(style='white', context='notebook', palette='muted')
import matplotlib.pyplot as plt

# 导入数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 分别查看实验数据集和预测数据集数据
# print('实验数据大小:', train.shape)
# print('预测数据大小:', test.shape)

# 将实验数据和预测数据合并
full = train.append(test, ignore_index=True)
full.describe()

# print(full.info())

sns.barplot(data=train, x='Embarked', y='Survived')
# plt.show()

# 计算不同类型embarked的乘客，其生存率为多少
# print('Embarked为"S"的乘客，其生存率为%.2f' % full['Survived'][full['Embarked'] == 'S'].value_counts(normalize=True)[1])
# print('Embarked为"C"的乘客，其生存率为%.2f' % full['Survived'][full['Embarked'] == 'C'].value_counts(normalize=True)[1])
# print('Embarked为"Q"的乘客，其生存率为%.2f' % full['Survived'][full['Embarked'] == 'Q'].value_counts(normalize=True)[1])

# 法国登船乘客生存率较高原因可能与其头等舱乘客比例较高有关
sns.catplot('Pclass', col='Embarked', data=train, kind='count', height=3)
# plt.show()

sns.barplot(data=train, x='Parch', y='Survived')
# plt.show()

sns.barplot(data=train, x='SibSp', y='Survived')
# plt.show()

sns.barplot(data=train, x='Pclass', y='Survived')
# plt.show()

sns.barplot(data=train, x='Sex', y='Survived')
# plt.show()

# 创建坐标轴
ageFacet = sns.FacetGrid(train, hue='Survived', aspect=3)
# 作图，选择图形类型
ageFacet.map(sns.kdeplot, 'Age', shade=True)
# 其他信息：坐标轴范围、标签等
ageFacet.set(xlim=(0, train['Age'].max()))
ageFacet.add_legend()
# plt.show()

# 创建坐标轴
ageFacet = sns.FacetGrid(train, hue='Survived', aspect=3)
ageFacet.map(sns.kdeplot, 'Fare', shade=True)
ageFacet.set(xlim=(0, 150))
ageFacet.add_legend()
# plt.show()

# 查看fare分布
farePlot = sns.distplot(full['Fare'][full['Fare'].notnull()], label='skewness:%.2f' % (full['Fare'].skew()))
farePlot.legend(loc='best')
# plt.show()

# 对数化处理fare值
full['Fare'] = full['Fare'].map(lambda x: np.log(x) if x > 0 else 0)

# 数据预处理
# 对Cabin缺失值进行处理，利用U（Unknown）填充缺失值
full['Cabin'] = full['Cabin'].fillna('U')
full['Cabin'].head()

# 对Embarked缺失值进行处理，查看缺失值情况
full[full['Embarked'].isnull()]
full['Embarked'].value_counts()
full['Embarked'] = full['Embarked'].fillna('S')

# 查看缺失数据情况，该乘客乘坐3等舱，登船港口为法国，舱位未知
full[full['Fare'].isnull()]
# 利用3等舱，登船港口为英国，舱位未知旅客的平均票价来填充缺失值
full['Fare'] = full['Fare'].fillna(
    full[(full['Pclass'] == 3) & (full['Embarked'] == 'S') & (full['Cabin'] == 'U')]['Fare'].mean())

# 特征工程
# 构造新特征Title
full['Title'] = full['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
# 查看title数据分布
full['Title'].value_counts()

# 将title信息进行整合
TitleDict = {}
TitleDict['Mr'] = 'Mr'
TitleDict['Mlle'] = 'Miss'
TitleDict['Miss'] = 'Miss'
TitleDict['Master'] = 'Master'
TitleDict['Jonkheer'] = 'Master'
TitleDict['Mme'] = 'Mrs'
TitleDict['Ms'] = 'Mrs'
TitleDict['Mrs'] = 'Mrs'
TitleDict['Don'] = 'Royalty'
TitleDict['Sir'] = 'Royalty'
TitleDict['the Countess'] = 'Royalty'
TitleDict['Dona'] = 'Royalty'
TitleDict['Lady'] = 'Royalty'
TitleDict['Capt'] = 'Officer'
TitleDict['Col'] = 'Officer'
TitleDict['Major'] = 'Officer'
TitleDict['Dr'] = 'Officer'
TitleDict['Rev'] = 'Officer'

full['Title'] = full['Title'].map(TitleDict)
full['Title'].value_counts()

# 可视化分析Title与Survived之间关系
sns.barplot(data=full, x='Title', y='Survived')
# plt.show()

full['familyNum'] = full['Parch'] + full['SibSp'] + 1
# 查看familyNum与Survived
sns.barplot(data=full, x='familyNum', y='Survived')


# plt.show()

# 我们按照家庭成员人数多少，将家庭规模分为“小、中、大”三类：
def familysize(familyNum):
    if familyNum == 1:
        return 0
    elif (familyNum >= 2) & (familyNum <= 4):
        return 1
    else:
        return 2


full['familySize'] = full['familyNum'].map(familysize)
full['familySize'].value_counts()

# 查看familySize与Survived
sns.barplot(data=full, x='familySize', y='Survived')
# plt.show()

# 提取Cabin字段首字母
full['Deck'] = full['Cabin'].map(lambda x: x[0])
# 查看不同Deck类型乘客的生存率
sns.barplot(data=full, x='Deck', y='Survived')
# plt.show()

# 提取各票号的乘客数量
TickCountDict = {}
TickCountDict = full['Ticket'].value_counts()
TickCountDict.head()

# 将同票号乘客数量数据并入数据集中
full['TickCot'] = full['Ticket'].map(TickCountDict)
full['TickCot'].head()

# 查看TickCot与Survived之间关系
sns.barplot(data=full, x='TickCot', y='Survived')


# plt.show()

# 按照TickCot大小，将TickGroup分为三类。
def TickCountGroup(num):
    if (num >= 2) & (num <= 4):
        return 0
    elif (num == 1) | ((num >= 5) & (num <= 8)):
        return 1
    else:
        return 2


# 得到各位乘客TickGroup的类别
full['TickGroup'] = full['TickCot'].map(TickCountGroup)
# 查看TickGroup与Survived之间关系
sns.barplot(data=full, x='TickGroup', y='Survived')

# 查看缺失值情况
full[full['Age'].isnull()].head()

# 筛选数据集
AgePre = full[['Age', 'Parch', 'Pclass', 'SibSp', 'Title', 'familyNum', 'TickCot']]
# 进行one-hot编码
AgePre = pd.get_dummies(AgePre)
ParAge = pd.get_dummies(AgePre['Parch'], prefix='Parch')
SibAge = pd.get_dummies(AgePre['SibSp'], prefix='SibSp')
PclAge = pd.get_dummies(AgePre['Pclass'], prefix='Pclass')
# 查看变量间相关性
AgeCorrDf = pd.DataFrame()
AgeCorrDf = AgePre.corr()
AgeCorrDf['Age'].sort_values()

# 拼接数据
AgePre = pd.concat([AgePre, ParAge, SibAge, PclAge], axis=1)
AgePre.head()

# 拆分实验集和预测集
AgeKnown = AgePre[AgePre['Age'].notnull()]
AgeUnKnown = AgePre[AgePre['Age'].isnull()]

# 生成实验数据的特征和标签
AgeKnown_X = AgeKnown.drop(['Age'], axis=1)
AgeKnown_y = AgeKnown['Age']
# 生成预测数据的特征
AgeUnKnown_X = AgeUnKnown.drop(['Age'], axis=1)

# 利用随机森林构建模型
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=None, n_estimators=500, n_jobs=-1)
rfr.fit(AgeKnown_X, AgeKnown_y)

# 模型得分
rfr.score(AgeKnown_X, AgeKnown_y)

# 预测年龄
AgeUnKnown_y = rfr.predict(AgeUnKnown_X)
# 填充预测数据
full.loc[full['Age'].isnull(), ['Age']] = AgeUnKnown_y
# print(full.info())  # 此时已无缺失值

# 提取乘客的姓氏及相应的乘客数
full['Surname'] = full['Name'].map(lambda x: x.split(',')[0].strip())
SurNameDict = {}
SurNameDict = full['Surname'].value_counts()
full['SurnameNum'] = full['Surname'].map(SurNameDict)

# 将数据分为两组
MaleDf = full[(full['Sex'] == 'male') & (full['Age'] > 12) & (full['familyNum'] >= 2)]
FemChildDf = full[((full['Sex'] == 'female') | (full['Age'] <= 12)) & (full['familyNum'] >= 2)]

#分析男性同组效应
MSurNamDf=MaleDf['Survived'].groupby(MaleDf['Surname']).mean()
MSurNamDf.head()
MSurNamDf.value_counts()

#获得生存率为1的姓氏
MSurNamDict={}
MSurNamDict=MSurNamDf[MSurNamDf.values==1].index
MSurNamDict

#分析女性及儿童同组效应
FCSurNamDf=FemChildDf['Survived'].groupby(FemChildDf['Surname']).mean()
FCSurNamDf.head()
FCSurNamDf.value_counts()

#获得生存率为0的姓氏
FCSurNamDict={}
FCSurNamDict=FCSurNamDf[FCSurNamDf.values==0].index
FCSurNamDict

#对数据集中这些姓氏的男性数据进行修正：1、性别改为女；2、年龄改为5。
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(MSurNamDict))&(full['Sex']=='male'),'Age']=5
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(MSurNamDict))&(full['Sex']=='male'),'Sex']='female'

#对数据集中这些姓氏的女性及儿童的数据进行修正：1、性别改为男；2、年龄改为60。
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(FCSurNamDict))&((full['Sex']=='female')|(full['Age']<=12)),'Age']=60
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(FCSurNamDict))&((full['Sex']=='female')|(full['Age']<=12)),'Sex']='male'

#人工筛选
fullSel=full.drop(['Cabin','Name','Ticket','PassengerId','Surname','SurnameNum'],axis=1)
#查看各特征与标签的相关性
corrDf=pd.DataFrame()
corrDf=fullSel.corr()
corrDf['Survived'].sort_values(ascending=True)

#热力图，查看Survived与其他特征间相关性大小
plt.figure(figsize=(8,8))
sns.heatmap(fullSel[['Survived','Age','Embarked','Fare','Parch','Pclass',
                    'Sex','SibSp','Title','familyNum','familySize','Deck',
                     'TickCot','TickGroup']].corr(),cmap='BrBG',annot=True,
           linewidths=.5)
plt.xticks(rotation=45)

fullSel=fullSel.drop(['familyNum','SibSp','TickCot','Parch'],axis=1)
#one-hot编码
fullSel=pd.get_dummies(fullSel)
PclassDf=pd.get_dummies(full['Pclass'],prefix='Pclass')
TickGroupDf=pd.get_dummies(full['TickGroup'],prefix='TickGroup')
familySizeDf=pd.get_dummies(full['familySize'],prefix='familySize')

fullSel=pd.concat([fullSel,PclassDf,TickGroupDf,familySizeDf],axis=1)

#构建模型
#拆分实验数据与预测数据
experData=fullSel[fullSel['Survived'].notnull()]
preData=fullSel[fullSel['Survived'].isnull()]

experData_X=experData.drop('Survived',axis=1)
experData_y=experData['Survived']
preData_X=preData.drop('Survived',axis=1)

#导入机器学习算法库
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold

if __name__ == "__main__":
    #设置kfold，交叉采样法拆分数据集
    kfold=StratifiedKFold(n_splits=10)

    #汇总不同模型算法
    classifiers=[]
    classifiers.append(SVC())
    classifiers.append(DecisionTreeClassifier())
    classifiers.append(RandomForestClassifier())
    classifiers.append(ExtraTreesClassifier())
    classifiers.append(GradientBoostingClassifier())
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression())
    classifiers.append(LinearDiscriminantAnalysis())

    #不同机器学习交叉验证结果汇总
    cv_results=[]
    for classifier in classifiers:
        cv_results.append(cross_val_score(classifier,experData_X,experData_y,
                                          scoring='accuracy',cv=kfold,n_jobs=-1))

    # 求出模型得分的均值和标准差
    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    # 汇总数据
    cvResDf = pd.DataFrame({'cv_mean': cv_means,
                            'cv_std': cv_std,
                            'algorithm': ['SVC', 'DecisionTreeCla', 'RandomForestCla', 'ExtraTreesCla',
                                          'GradientBoostingCla', 'KNN', 'LR', 'LinearDiscrimiAna']})

    cvResDf

    # sns.barplot(data=cvResDf,x='cv_mean',y='algorithm',**{'xerr':cv_std})

    cvResFacet=sns.FacetGrid(cvResDf.sort_values(by='cv_mean',ascending=False),sharex=False,
                sharey=False,aspect=2)
    cvResFacet.map(sns.barplot,'cv_mean','algorithm',**{'xerr':cv_std},
                   palette='muted')
    cvResFacet.set(xlim=(0.7,0.9))
    cvResFacet.add_legend()
    plt.show()
    #GradientBoostingClassifier模型
    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss' : ["deviance"],
                  'n_estimators' : [100,200,300],
                  'learning_rate': [0.1, 0.05, 0.01],
                  'max_depth': [4, 8],
                  'min_samples_leaf': [100,150],
                  'max_features': [0.3, 0.1]
                  }
    modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold,
                                         scoring="accuracy", n_jobs= -1, verbose = 1)
    modelgsGBC.fit(experData_X,experData_y)

    #LogisticRegression模型
    modelLR=LogisticRegression()
    LR_param_grid = {'C' : [1,2,3],
                    'penalty':['l1','l2']}
    modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold,
                                         scoring="accuracy", n_jobs= -1, verbose = 1)
    modelgsLR.fit(experData_X,experData_y)

    #modelgsGBC模型
    print('modelgsGBC模型得分为：%.3f'%modelgsGBC.best_score_)
    #modelgsLR模型
    print('modelgsLR模型得分为：%.3f'%modelgsLR.best_score_)

    #模型预测
    #TitanicGBSmodle
    GBCpreData_y=modelgsGBC.predict(preData_X)
    GBCpreData_y=GBCpreData_y.astype(int)
    #导出预测结果
    GBCpreResultDf=pd.DataFrame()
    GBCpreResultDf['PassengerId']=full['PassengerId'][full['Survived'].isnull()]
    GBCpreResultDf['Survived']=GBCpreData_y
    GBCpreResultDf
    #将预测结果导出为csv文件
    GBCpreResultDf.to_csv('TitanicGBSmodle.csv',index=False)