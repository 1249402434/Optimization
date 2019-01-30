from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np

clfs=[RandomForestClassifier(n_estimators=10,max_depth=3),GradientBoostingClassifier(n_estimators=10),SVC(probability=True)]

data, target = make_blobs(n_samples=50000, centers=2, random_state=0, cluster_std=0.60)
#整个数据集分为训练测试
train_x,test_x,train_y,test_y=train_test_split(data,target,test_size=0.25,random_state=2019)
#训练集分为不相重叠的两部分
data1_x,data2_x,data1_y,data2_y=train_test_split(train_x,train_y,test_size=0.5)
#存放每个分类器对第二部分训练集的预测结果
data_blend=np.zeros((len(data2_x),len(clfs)))
#存放对测试集的预测结果
test_blend=np.zeros((len(test_x),len(clfs)))
for i,clf in enumerate(clfs):
    clf.fit(data1_x,data1_y)
    y=clf.predict_proba(data2_x)[:,1]
    data_blend[:,i]=y
    test_blend[:,i]=clf.predict_proba(test_x)[:,1]


lr=LogisticRegression()
lr.fit(data_blend,data2_y)
print(lr.score(test_blend,test_y))

y_submission=lr.predict_proba(test_blend)[:,1]
y_submission=(y_submission-y_submission.min())/(y_submission.max()-y_submission.min())

print('val auc score: ',roc_auc_score(test_y,y_submission))


