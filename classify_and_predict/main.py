import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  cross_val_score
from sklearn.preprocessing import LabelEncoder

iris = pd.read_csv('iris.csv')
#rs = iris.sample(5)#随机从文件中取5组样本
#print(rs)

#对花瓣的长度和宽度作回归分析图
sns.regplot(x='PetalLengthCm', y='PetalWidthCm',data=iris)
#plt.show()

#采用DataFrame的map函数，利用字典映射-把Species替换为123
Species_dict={
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
y = iris.Species.map(Species_dict)
#print(y1)

#构造模型
model_log = linear_model.LogisticRegression()
features=['PetalLengthCm','PetalWidthCm','SepalLengthCm','SepalWidthCm']
X = iris[features]
model_log.fit(X,y)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
#评估模型
features=['PetalLengthCm']
X = iris[features]
scores = cross_val_score(model_log, X, y, cv=5, scoring='accuracy')    #accuracy表示属于某个类别的准确率

print(np.mean(scores))