import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('train_local.csv')
print(train_data.head())
test_data = pd.read_csv('test_local.csv')
print(test_data.head())
### 训练集特征说明
# - PassengerID (ID)
# - Survived (是否存活)
# - Pclass (客舱等级，重要)
# - Name (姓名，可结合爬虫)
# - Sex (性别，重要)
# - Age (年龄，重要)
# - SibSp (旁系亲友)
# - Parch (直系亲属)
# - Ticket (票编号)
# - Fare (票价)
# - Cabin (客舱编号)
# - Embarked (上船港口编号)
all_data = pd.concat([train_data, test_data], ignore_index=True, sort=True)

print(train_data['Survived'].value_counts())
