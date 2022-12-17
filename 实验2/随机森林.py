### 先初始化几个参数拟合尝试下
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score, make_scorer, f1_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from collections import Counter
from 决策树_预处理 import pre

df = pre()
print('----------建立随机森林模型实现预测部分----------')
x_cols = [col for col in df.columns if col != 'Target']
y_col = 'Target'

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

#尝试使用100颗树，后续根据需要再降低树的数量
rf_model = RandomForestClassifier(class_weight='balanced',random_state=10)
rf_model.fit(X_train_std, y_train)
y_test_pred3 = rf_model.predict(X_test_std)
print('accuracy of rf_model score',accuracy_score(y_test, y_test_pred3))
print('f1_score of rf_model score',f1_score(y_test, y_test_pred3))
print('recall_score of rf_model score',recall_score(y_test, y_test_pred3))
print('precision_score of rf_model score',precision_score(y_test, y_test_pred3))

### 调参
### 对class_weight进行搜索
param_test0 = {'class_weight':[{0:1,1:3},{0:1,1:5},{0:1,1:10},{0:1,1:20},'balanced']}
gsearch0 = GridSearchCV(
    estimator = rf_model,
    param_grid = param_test0,
    cv=5
    )
gsearch0.fit(X_train_std,y_train)
print('best params:{}'.format(gsearch0.best_params_))
print('best score:{}'.format(gsearch0.best_score_))

### 对n_estimators进行搜索
param_test1 = {'n_estimators':range(10,101,10)}
gsearch1 = GridSearchCV(
    estimator = gsearch0.best_estimator_,
    param_grid = param_test1,
    cv=5
    )
gsearch1.fit(X_train_std,y_train)
print('best params:{}'.format(gsearch1.best_params_))
print('best score:{}'.format(gsearch1.best_score_))

# 我们得到了最佳的决策树数，目前的随机森林有过拟合倾向，因为叶子最小样本数、最小分割样本数、最大深度等都是None，没有剪枝操作。
### 比较它与决策树在测试集上的效果
y_test_final = gsearch1.best_estimator_.predict(X_test_std)
print('accuracy of rf_model score',accuracy_score(y_test, y_test_final))
print('f1_score of rf_model score',f1_score(y_test, y_test_final))
print('recall_score of rf_model score',recall_score(y_test,y_test_final))
print('precision_score of rf_model score',precision_score(y_test, y_test_final))

