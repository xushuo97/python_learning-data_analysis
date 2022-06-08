#利用python实现对鸢尾花数据降维分析-2022年6月8日-代码来源自github
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 加载数据
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('iris.csv')
# 打印数据信息
print('Species类别、数据size、数据前三行、各个类别的数量：')
print(set(data['Species'].values))
print(data.shape)
print(data.head(3))# head默认无填充为5
print(data['Species'].value_counts())

# 可视化维度关系
values = data.iloc[:, 1:5] # 取四列数值
correlation = values.corr() # 列与列之间的相关系数
fig, ax = plt.subplots(figsize=(8, 8))

val_std = StandardScaler().fit_transform(values)#对数据进行标准化处理
print('标准化之后的数据：')
#print(val_std)

#热力图
sns.heatmap(correlation, annot=True, annot_kws={'size': 10}, cmap='Reds', square=True, ax=ax)
plt.show()

#散点关系图
sns.pairplot(data.iloc[:, 1:6], hue='Species')
plt.show()

#使用PCA分析数据集（4个主成分）
pca4 = PCA(n_components=4)
pc4 = pca4.fit_transform(val_std) # fit_transform(X) 用X来训练PCA模型，同时返回降维后的数据,这里的pca4就是降维后的数据

X=pca4.inverse_transform(pc4)#转换成原始数据
print('将降维后的pca4转换为原始的数据并输出：')
#print(X)

#各个主成分所占的比例
print('pca4降维分析中各个成分所占的比例')
print('explained variance ratio:%s' % pca4.explained_variance_ratio_)
# explained_variance_ratio_：返回所保留的n个成分各自的方差百分比。
com_p=pca4.components_#components_ ：返回具有最大方差的成分。方差越大，表明包含的“信息”就越多
print('将标准数据线性变换到四维的线性变换系数')
print(com_p)


#可视化主成分累加结果
plt.plot(range(1, 5), np.cumsum(pca4.explained_variance_ratio_))
plt.scatter(range(1, 5), np.cumsum(pca4.explained_variance_ratio_))
plt.xlim(0,5)
plt.ylim(0.9,1.02)
plt.xlabel('numbere of components')
plt.ylabel('cumsum explained variance')
plt.show()

#2个主成分
pca2 = PCA(n_components=2)
pc2 = pca2.fit_transform(val_std)# fit_transform(X) 用X来训练PCA模型，同时返回降维后的数据

#各个主成分所占的比例
print('pca2降维分析中各个成分所占的比例')
print('explained variance ratio:%s' % pca2.explained_variance_ratio_)
Y=pca2.inverse_transform(pc2)#转换成原始数据
print('将降维后的pca2转换为原始的数据并输出：')
#print(Y)

pc2_df = pd.DataFrame(pc2,columns=['pc1', 'pc2'])#列名
pc2_df['Species'] = data['Species']
print(pc2_df)
#2个主成分内容
comp=pca2.components_#components_ ：返回具有最大方差的成分。方差越大，表明包含的“信息”就越多
print('将标准数据线性变换到两维的线性变换系数')
print(comp)


#验证系数对原始数据的表示
result=np.dot(val_std[0],comp[0]), np.dot(val_std[0],comp[1])
print('数据验证,结果与pc1、pc2的数据一致即正确')
print(result)

#数据从4维降到2维，绘制二维图

#提取各个物种对应的主成分数据
setosa=pc2_df[pc2_df['Species']=='Iris-setosa']# print(setosa)
virginica=pc2_df[pc2_df['Species']=='Iris-virginica']
versicolor=pc2_df[pc2_df['Species']=='Iris-versicolor']

fig,ax=plt.subplots(figsize=(8,8))

plt.scatter(setosa['pc1'],setosa['pc2'],alpha=0.5,color='red',label='Iris-setosa')
plt.scatter(virginica['pc1'],virginica['pc2'],alpha=0.5,color='green',label='Iris-virginica')
plt.scatter(versicolor['pc1'],versicolor['pc2'],alpha=0.5,color='blue',label='Iris-versicolor')

plt.legend(loc='best')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()