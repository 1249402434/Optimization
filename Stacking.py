'''
数据量较小，所以效果不明显
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

data=load_iris()
X,y=data.data,data.target
#首先将整体数据集划分为训练集和测试集
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.25)
#第一轮的多个分类器
clfs=[LogisticRegression(),SVC(),GradientBoostingClassifier(n_estimators=10)]
#对训练集做5折划分，4折用于训练，1折用于测试
skf=KFold(n_splits=5)

train_stack=np.zeros((len(train_x),len(clfs)))
test_stack=np.zeros((len(test_x),5))
test_stack_model=np.zeros((len(test_x),len(clfs)))

for i,clf in enumerate(clfs):
    for j,(train_index,test_index) in enumerate(skf.split(train_x,train_y)):
        print('train_index:',train_index)
        print('test_index:',test_index)
        clf.fit(train_x[train_index],train_y[train_index])
        train_stack[test_index,i]=clf.predict(train_x[test_index])
        test_stack[:,j]=clf.predict(test_x)
    test_stack_model[:,i]=test_stack.mean(1)


randforest=RandomForestClassifier(n_estimators=10,max_depth=3,random_state=2019)
randforest.fit(train_stack,train_y)
print(randforest.score(test_stack_model,test_y))










