from __future__ import print_function
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

#从文件中读取坐标点
result = []
with open('k-means_data.txt', 'r') as f:
    for line in f:
        x = list(map(float,(line.strip('\n').split(','))))
        result.append(x)
dataSet = mat(result)

# 计算两个点的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 为给定数据集构建一个包含 k 个随机质心的集合并且随机生成k个质心
def randCent(dataMat, k):
    n = shape(dataMat)[1]
    centroids = mat(zeros((k, n)))  # 创建k个质心矩阵,k行,n列
    for j in range(n):
        minJ = min(dataMat[:, j])
        rangeJ = float(max(dataMat[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids

# 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。这个过程重复数次，直到数据点的簇分配结果不再改变位置。
def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros(
        (m, 2)))  # 创建一个与 dataMat 行数一样，但是有两列的矩阵，用来保存簇分配结果
    centroids = createCent(dataMat, k)  # 创建质心，随机k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :],
                                  dataMat[i, :])
                if distJI < minDist:  # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[
                    i, :] = minIndex, minDist**2
        # print(centroids)
        for cent in range(k):  # 更新质心
            ptsInClust = dataMat[nonzero(
                clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(
                ptsInClust, axis=0)
    return centroids, clusterAssment

a,b = kMeans(dataSet,4)
# print(b[:,0])
d = array(b[:,0].tolist())
# print(d)
# e = zeros(len(d))
e = []
for x in range(len(d)):
    # e = append(e, d[x][0])
    e.append(int(d[x][0]))
print("e:",e)

x = dataSet[:,0]
y = dataSet[:,1]
z = dataSet[:,2]

x1 = a[:,0]
y1 = a[:,1]
z1 = a[:,2]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x,y,z,c = e,label='数据点')
ax.scatter(x1,y1,z1,c='r',marker="*",label='簇点')

ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()