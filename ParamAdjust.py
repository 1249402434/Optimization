from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier

iris=datasets.load_iris()
X, y= iris.data, iris.target
forest=RandomForestClassifier()
param_list={
    'criterion':['gini','entropy'],
    'max_features':range(1,4),
    'max_depth':range(3,5),
    'min_samples_split':range(2,10),
    'min_samples_leaf':range(1,10)
}
'''
n_iter进行随机参数搜索的次数
cv几折交叉验证
'''
random=RandomizedSearchCV(forest,param_list,n_iter=20,scoring='accuracy',cv=3)
random.fit(X,y)
print(random.best_params_)
print(random.best_score_)


grid=GridSearchCV(forest,param_grid=param_list,scoring='accuracy')
grid.fit(X,y)
print(grid.best_score_)
print(grid.best_params_)